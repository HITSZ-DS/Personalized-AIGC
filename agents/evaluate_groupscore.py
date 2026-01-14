"""
GroupScore Evaluation for Generated Content

This script evaluates the overall consistency of generated content:
1. Image-Text Consistency (Redbook/ITProduct): Using CLIP model
2. Link-Text Relevance (Hupu/CommentProduct): Using text embeddings

GroupScore is calculated as the aggregated similarity score across all pairs,
measuring how well the content components align with their context.
"""

import os
import re
import json
import argparse
import requests
import hashlib
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Lazy imports for optional dependencies
# torch and CLIP will be imported only when CLIPEvaluator is initialized
# PIL.Image will be imported only when image operations are needed

@dataclass
class ImageTextPair:
    """Represents an image-text pair extracted from HTML."""
    image_path: str
    image_name: str
    preceding_text: str
    following_text: str
    combined_text: str
    caption: Optional[str] = None


@dataclass
class PairScore:
    """Score for a single image-text pair."""
    image_name: str
    preceding_score: float
    following_score: float
    combined_score: float
    caption_score: Optional[float] = None


@dataclass
class LinkTextPair:
    """Represents a link-text pair extracted from HTML."""
    link_title: str
    link_url: str
    link_platform: str
    preceding_text: str
    following_text: str
    combined_text: str


@dataclass
class LinkScore:
    """Score for a single link-text pair."""
    link_title: str
    link_platform: str
    preceding_score: float
    following_score: float
    combined_score: float


@dataclass
class GroupScoreResult:
    """Result of GroupScore evaluation.
    
    Thread Safety: Each thread gets its own independent result instance.
    The file_path field ensures correct product-score mapping in multi-threaded scenarios.
    """
    file_name: str
    file_path: Optional[str] = None  # Full path for thread-safety verification
    evaluation_type: str = "image-text"  # "image-text" or "link-text"
    num_pairs: int = 0
    pair_scores: List[Dict] = None
    group_score_mean: float = 0.0
    group_score_harmonic: float = 0.0
    group_score_min: float = 0.0
    group_score_geometric: float = 0.0
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.pair_scores is None:
            self.pair_scores = []


class TextEmbeddingEvaluator:
    """Text Embedding-based Link-Text Relevance Evaluator with caching and parallelization."""
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the text embedding evaluator.
        
        Args:
            api_key: API key for embedding service
            api_base: API base URL
            model: Embedding model name
        """
        self.api_key = api_key or os.getenv("SEARCH_API_KEY", "")
        self.api_base = api_base or os.getenv("SEARCH_BASE_URL", "https://yunwu.ai/v1")
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not self.api_key:
            raise ValueError("API key not found. Set SEARCH_API_KEY environment variable.")
        
        # Embedding cache: key -> embedding vector
        self._embedding_cache = {}
        self._cache_lock = threading.Lock()
        
        print(f"Using Text Embedding model: {self.model} (with caching and parallelization)")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Normalize text: strip and truncate
        normalized = text.strip()[:8000]
        # Use hash for cache key
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text (with caching).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text.strip():
            # Return zero vector for empty text
            return np.zeros(1536)  # Default embedding dimension
        
        # Check cache
        cache_key = self._get_cache_key(text)
        with self._cache_lock:
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key].copy()
        
        # Cache miss, fetch from API
        try:
            response = requests.post(
                f"{self.api_base}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": text[:8000]  # Truncate to avoid token limits
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = np.array(data['data'][0]['embedding'])
                # Store in cache
                with self._cache_lock:
                    self._embedding_cache[cache_key] = embedding.copy()
                return embedding
            else:
                print(f"⚠️  Embedding API error: {response.status_code}")
                return np.zeros(1536)
        except Exception as e:
            print(f"⚠️  Embedding request failed: {e}")
            return np.zeros(1536)
    
    def get_embeddings_batch(self, texts: List[str], max_workers: int = 5) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts in parallel (with caching).
        
        Args:
            texts: List of texts to embed
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Separate cached and uncached texts
        cached_results = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if not text.strip():
                cached_results[i] = np.zeros(1536)
                continue
            
            cache_key = self._get_cache_key(text)
            with self._cache_lock:
                if cache_key in self._embedding_cache:
                    cached_results[i] = self._embedding_cache[cache_key].copy()
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        
        # Fetch uncached texts in parallel
        if uncached_texts:
            embeddings_list = [None] * len(uncached_texts)
            
            def fetch_embedding(idx, text):
                return idx, self.get_embedding(text)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(fetch_embedding, i, text): i 
                          for i, text in enumerate(uncached_texts)}
                
                for future in as_completed(futures):
                    orig_idx = futures[future]
                    try:
                        _, embedding = future.result()
                        embeddings_list[orig_idx] = embedding
                    except Exception as e:
                        print(f"⚠️  Parallel embedding fetch failed: {e}")
                        embeddings_list[orig_idx] = np.zeros(1536)
            
            # Store uncached results
            for i, orig_idx in enumerate(uncached_indices):
                cached_results[orig_idx] = embeddings_list[i]
        
        # Return in original order
        return [cached_results[i] for i in range(len(texts))]
    
    def clear_cache(self):
        """Clear the embedding cache."""
        with self._cache_lock:
            self._embedding_cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                "cache_size": len(self._embedding_cache),
                "cache_keys": list(self._embedding_cache.keys())[:10]  # First 10 keys for debugging
            }
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text (e.g., link title)
            text2: Second text (e.g., context)
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1.strip() or not text2.strip():
            return 0.0
        
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Normalize vectors
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Convert from [-1, 1] to [0, 1] range
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def compute_similarities_batch(self, text_pairs: List[Tuple[str, str]], max_workers: int = 5) -> List[float]:
        """
        Compute cosine similarities for multiple text pairs in parallel.
        
        Args:
            text_pairs: List of (text1, text2) tuples
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of similarity scores
        """
        if not text_pairs:
            return []
        
        # Collect all unique texts
        all_texts = []
        pair_indices = []
        for text1, text2 in text_pairs:
            if not text1.strip() or not text2.strip():
                pair_indices.append((None, None))
                continue
            idx1 = len(all_texts)
            all_texts.append(text1)
            idx2 = len(all_texts)
            all_texts.append(text2)
            pair_indices.append((idx1, idx2))
        
        # Get embeddings in parallel
        embeddings = self.get_embeddings_batch(all_texts, max_workers=max_workers)
        
        # Compute similarities
        similarities = []
        for idx1, idx2 in pair_indices:
            if idx1 is None or idx2 is None:
                similarities.append(0.0)
                continue
            
            emb1 = embeddings[idx1]
            emb2 = embeddings[idx2]
            
            # Normalize vectors
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # Compute cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            similarity = (similarity + 1) / 2  # Convert to [0, 1]
            
            similarities.append(float(similarity))
        
        return similarities


class CLIPEvaluator:
    """
    CLIP-based Image-Text Consistency Evaluator with singleton pattern for model sharing.
    
    This class uses a singleton pattern to ensure that only one model instance is loaded
    per (model_name, device) combination, even when multiple threads create CLIPEvaluator
    instances. This prevents memory explosion when processing multiple users in parallel.
    
    Thread Safety:
    - Model loading is protected by a lock (double-checked locking pattern)
    - Model inference is thread-safe (PyTorch models are safe for concurrent read access)
    - All threads share the same model instance, reducing memory usage from ~40x to 1x
    """
    
    # Class-level cache for model instances (singleton pattern)
    # Key format: "model_name:device" (e.g., "ViT-B/32:cuda")
    _instances = {}
    _lock = None  # Will be initialized when threading is imported
    
    def __new__(cls, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Singleton pattern: return existing instance if available, otherwise create new one.
        
        Args:
            model_name: CLIP model variant to use
            device: Device to run the model on (cuda/cpu/auto)
        """
        # Initialize lock on first use
        if cls._lock is None:
            import threading
            cls._lock = threading.Lock()
        
        # Determine actual device for cache key
        # We need to check CUDA availability to create proper cache key
        try:
            import torch
            if device is None or device == "auto":
                actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                actual_device = device
        except ImportError:
            # If torch not available, use device as-is or default to cpu
            actual_device = device if device and device != "auto" else "cpu"
        
        # Create cache key from model_name and actual device
        cache_key = f"{model_name}:{actual_device}"
        
        # Check if instance exists (double-checked locking pattern)
        if cache_key not in cls._instances:
            with cls._lock:
                # Check again after acquiring lock (double-checked locking)
                if cache_key not in cls._instances:
                    # Create new instance
                    instance = super(CLIPEvaluator, cls).__new__(cls)
                    cls._instances[cache_key] = instance
                    # Don't set _initialized here - let __init__ handle it
                    instance._cache_key = cache_key
                    return instance
                else:
                    # Another thread created it while we were waiting
                    return cls._instances[cache_key]
        else:
            # Instance already exists, return it
            return cls._instances[cache_key]
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize the CLIP evaluator (only called once per cache key due to singleton).
        
        CRITICAL: In Python, __init__ is ALWAYS called even if __new__ returns an existing instance.
        We use a lock to ensure only the first thread loads the model.

        Args:
            model_name: CLIP model variant to use
            device: Device to run the model on (cuda/cpu/auto)
        """
        # CRITICAL: Check if already initialized first (fast path)
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # Use lock to ensure only one thread loads the model
        with self._lock:
            # Double-check after acquiring lock (another thread might have initialized it)
            if hasattr(self, '_initialized') and self._initialized:
                return
            
            # Import dependencies only when needed
            try:
                import torch
                from PIL import Image
                import open_clip as clip
                self.torch = torch
                self.Image = Image
                self.clip = clip
                self.CLIP_BACKEND = "open_clip"
            except ImportError as e:
                raise ImportError(
                    f"CLIP dependencies not installed: {e}\n"
                    "Please install: pip install torch open-clip-torch pillow"
                )
            
            if device is None or device == "auto":
                self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            import threading
            thread_id = threading.current_thread().ident
            print(f"[Thread {thread_id}] Loading CLIP model ({model_name}) on {self.device}...")

            self.model, _, self.preprocess = self.clip.create_model_and_transforms(
                model_name.replace("/", "-"), pretrained='openai'
            )
            self.model = self.model.to(self.device)
            self.tokenize = self.clip.get_tokenizer(model_name.replace("/", "-"))

            self.model.eval()
            print(f"[Thread {thread_id}] CLIP model loaded successfully! (Instance ID: {id(self)})")
            
            # Mark as initialized BEFORE releasing lock (critical!)
            self._initialized = True

    def encode_image(self, image_path: str):
        """
        Encode an image into CLIP embedding space.
        
        Thread-safe: Each thread creates its own input tensor and gets independent results.

        Args:
            image_path: Path to the image file

        Returns:
            Normalized image embedding tensor
        """
        # Thread-safe: Each thread creates its own image tensor
        image = self.Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Thread-safe: PyTorch model inference is safe for concurrent read access
        # Each thread gets independent results based on its own input
        with self.torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def encode_text(self, text: str, max_length: int = 77):
        """
        Encode text into CLIP embedding space.
        
        Thread-safe: Each thread creates its own input tensor and gets independent results.

        Args:
            text: Text to encode
            max_length: Maximum token length (CLIP default is 77)

        Returns:
            Normalized text embedding tensor
        """
        # Truncate text if too long for CLIP
        # Note: combined_text is already truncated in extract_image_text_pairs to prioritize caption
        # This truncation is a safety measure for edge cases
        text = text[:500]  # Rough character limit before tokenization

        # Thread-safe: Each thread creates its own token tensor
        text_tokens = self.tokenize([text]).to(self.device)

        # Thread-safe: PyTorch model inference is safe for concurrent read access
        # Each thread gets independent results based on its own input
        with self.torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def compute_similarity(self, image_path: str, text: str) -> float:
        """
        Compute cosine similarity between an image and text.
        
        Thread-safe: This method is thread-safe because:
        1. Each thread calls encode_image/encode_text with its own inputs
        2. Each thread gets independent tensor results
        3. The dot product computation is independent for each thread
        4. The model is in eval() mode (read-only)
        
        Each thread's computation is completely independent, ensuring scores
        are correctly associated with their respective products.

        Args:
            image_path: Path to the image file
            text: Text to compare with the image

        Returns:
            Similarity score between 0 and 1
        """
        if not text.strip():
            return 0.0

        # Thread-safe: Each thread gets independent features for its own image/text
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(text)

        # Thread-safe: Each thread computes its own similarity independently
        similarity = (image_features @ text_features.T).item()
        # Convert from [-1, 1] to [0, 1] range
        similarity = (similarity + 1) / 2

        return similarity


class HTMLParser:
    """Parser for extracting image-text pairs from HTML files."""

    def __init__(self, html_path: str):
        """
        Initialize the HTML parser.

        Args:
            html_path: Path to the HTML file
        """
        self.html_path = Path(html_path)
        self.base_dir = self.html_path.parent

        with open(html_path, 'r', encoding='utf-8') as f:
            self.soup = BeautifulSoup(f.read(), 'html.parser')

    def extract_image_text_pairs(self) -> List[ImageTextPair]:
        """
        Extract all image-text pairs from the HTML.

        Returns:
            List of ImageTextPair objects
        """
        pairs = []

        # Find all image containers
        image_divs = self.soup.find_all('div', class_='post-image')

        for img_div in image_divs:
            img_tag = img_div.find('img')
            if not img_tag:
                continue

            img_src = img_tag.get('src', '')
            if not img_src:
                continue

            image_path = self.base_dir / img_src
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue

            # Get caption if available
            # 优先从data-caption-en属性读取英文caption（用于CLIP计算）
            # 如果没有，则从文本内容读取（向后兼容）
            caption_div = img_div.find('div', class_='image-caption')
            caption = None
            if caption_div:
                # 优先使用data-caption-en属性（英文，用于CLIP）
                caption = caption_div.get('data-caption-en')
                if not caption:
                    # 降级：从文本内容读取（可能是中文或英文）
                    caption = caption_div.get_text(strip=True)

            # Get preceding paragraph text
            preceding_text = self._get_preceding_text(img_div)

            # Get following paragraph text
            following_text = self._get_following_text(img_div)

            # Combine texts for overall context
            # 重要：如果有caption，将其包含在combined_text中
            # 这样caption会作为上下文的一部分参与评分
            # 方案2：如果文本过长，优先保留caption，截断其他部分
            MAX_TEXT_LENGTH = 300  # CLIP文本长度限制
            
            if caption:
                caption_len = len(caption)
                remaining_len = MAX_TEXT_LENGTH - caption_len - 2  # -2 for spaces
                
                if remaining_len > 0:
                    # 优先保留caption，然后按比例分配preceding和following
                    # 如果preceding和following都很长，各取一半剩余长度
                    preceding_len = len(preceding_text)
                    following_len = len(following_text)
                    total_context_len = preceding_len + following_len
                    
                    if total_context_len <= remaining_len:
                        # 如果总长度不超过限制，全部保留
                        combined_text = f"{preceding_text} {caption} {following_text}".strip()
                    else:
                        # 需要截断，优先保留caption，然后按比例截断preceding和following
                        # 分配策略：如果preceding和following都存在，各取一半；否则全部给存在的那个
                        if preceding_len > 0 and following_len > 0:
                            # 各取一半剩余长度
                            each_len = remaining_len // 2
                            truncated_preceding = preceding_text[:each_len] if preceding_len > each_len else preceding_text
                            truncated_following = following_text[:each_len] if following_len > each_len else following_text
                        elif preceding_len > 0:
                            # 只有preceding，全部剩余长度给它
                            truncated_preceding = preceding_text[:remaining_len] if preceding_len > remaining_len else preceding_text
                            truncated_following = ""
                        else:
                            # 只有following，全部剩余长度给它
                            truncated_preceding = ""
                            truncated_following = following_text[:remaining_len] if following_len > remaining_len else following_text
                        
                        combined_text = f"{truncated_preceding} {caption} {truncated_following}".strip()
                else:
                    # caption本身就已经超过或接近限制，只使用caption
                    combined_text = caption[:MAX_TEXT_LENGTH]
            else:
                # 没有caption，正常拼接并截断
                combined_text = f"{preceding_text} {following_text}".strip()
                if len(combined_text) > MAX_TEXT_LENGTH:
                    combined_text = combined_text[:MAX_TEXT_LENGTH]

            pairs.append(ImageTextPair(
                image_path=str(image_path),
                image_name=img_src,
                preceding_text=preceding_text,
                following_text=following_text,
                combined_text=combined_text,
                caption=caption
            ))

        return pairs

    def _get_preceding_text(self, element, max_distance=5) -> str:
        """Get text from the preceding paragraph element, skipping non-text elements."""
        current = element
        for _ in range(max_distance):
            prev_sibling = current.find_previous_sibling()
            if not prev_sibling:
                break
            # Accept paragraph or heading elements
            if prev_sibling.name in ['p', 'h2', 'h3']:
                text = self._clean_text(prev_sibling.get_text())
                if text:  # Only return non-empty text
                    return text
            current = prev_sibling
        return ""

    def _get_following_text(self, element, max_distance=5) -> str:
        """Get text from the following paragraph element, skipping non-text elements."""
        current = element
        for _ in range(max_distance):
            next_sibling = current.find_next_sibling()
            if not next_sibling:
                break
            # Accept paragraph or heading elements
            if next_sibling.name in ['p', 'h2', 'h3']:
                text = self._clean_text(next_sibling.get_text())
                if text:  # Only return non-empty text
                    return text
            current = next_sibling
        return ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_full_post_content(self) -> str:
        """Get the full text content of the post."""
        content_div = self.soup.find('div', class_='post-content')
        if content_div:
            return self._clean_text(content_div.get_text())
        return ""

    def get_tags(self) -> List[str]:
        """Extract hashtags from the post."""
        tags = []
        tag_div = self.soup.find('div', class_='post-tags')
        if tag_div:
            for tag_span in tag_div.find_all('span', class_='tag'):
                tags.append(tag_span.get_text(strip=True))
        return tags
    
    def extract_content_sequence(self) -> List[Dict]:
        """
        Extract content sequence (text, image, link order) for layout evaluation.
        
        Returns:
            List of content elements in order with their types
        """
        sequence = []
        content_div = self.soup.find('div', class_='post-content')
        if not content_div:
            return sequence
        
        text_idx = 0
        image_idx = 0
        link_idx = 0
        
        for element in content_div.children:
            if element.name == 'p' or element.name == 'h2':
                text_content = element.get_text(strip=True)
                if text_content:
                    sequence.append({
                        "type": "text",
                        "index": text_idx,
                        "content": text_content,
                        "char_count": len(text_content)
                    })
                    text_idx += 1
            
            elif element.name == 'div' and 'post-image' in element.get('class', []):
                img_tag = element.find('img')
                if img_tag and img_tag.get('src'):
                    sequence.append({
                        "type": "image",
                        "index": image_idx,
                        "src": img_tag['src']
                    })
                    image_idx += 1
            
            elif element.name == 'a' and 'link-card' in element.get('class', []):
                title_tag = element.find('div', class_='link-title')
                platform_tag = element.find('span', class_='link-platform-tag')
                
                sequence.append({
                    "type": "link",
                    "index": link_idx,
                    "title": title_tag.get_text(strip=True) if title_tag else "",
                    "platform": platform_tag.get_text(strip=True) if platform_tag else ""
                })
                link_idx += 1
        
        return sequence
    
    def extract_link_text_pairs(self) -> List[LinkTextPair]:
        """
        Extract all link-text pairs from the HTML (for Hupu posts).
        
        Returns:
            List of LinkTextPair objects
        """
        pairs = []
        
        # Find all link cards
        link_cards = self.soup.find_all('a', class_='link-card')
        
        for link_card in link_cards:
            # Get link information
            title_tag = link_card.find('div', class_='link-title')
            platform_tag = link_card.find('span', class_='link-platform-tag')
            
            if not title_tag:
                continue
            
            link_title = title_tag.get_text(strip=True)
            link_url = link_card.get('href', '#')
            link_platform = platform_tag.get_text(strip=True) if platform_tag else ""
            
            # Get preceding paragraph text
            preceding_text = self._get_preceding_text(link_card)
            
            # Get following paragraph text
            following_text = self._get_following_text(link_card)
            
            # Combine texts for overall context
            combined_text = f"{preceding_text} {following_text}".strip()
            
            pairs.append(LinkTextPair(
                link_title=link_title,
                link_url=link_url,
                link_platform=link_platform,
                preceding_text=preceding_text,
                following_text=following_text,
                combined_text=combined_text
            ))
        
        return pairs


def _ensure_english_caption(caption: str) -> str:
    """
    确保caption是英文，如果是中文则翻译
    
    Args:
        caption: 原始caption（可能是中文或英文）
        
    Returns:
        英文caption
    """
    if not caption or not caption.strip():
        return caption
    
    # 检测是否包含中文字符
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', caption))
    
    if not has_chinese:
        # 已经是英文，直接返回
        return caption
    
    # 包含中文，需要翻译
    # 使用简单的翻译映射表（常见关键词）
    translation_map = {
        "抹茶饮品": "matcha drink",
        "拿铁咖啡": "latte coffee",
        "咖啡店": "coffee shop",
        "好物分享": "product review",
        "旅行vlog": "travel vlog",
        "生活分享": "lifestyle",
        "美食": "food",
        "旅行": "travel",
        "咖啡": "coffee",
        "探店": "cafe visit",
        "产品": "product",
        "分享": "share",
        "推荐": "recommendation",
        "测评": "review"
    }
    
    # 尝试直接匹配
    if caption in translation_map:
        return translation_map[caption]
    
    # 尝试部分匹配
    for chinese, english in translation_map.items():
        if chinese in caption:
            # 如果caption包含映射表中的中文词，尝试替换
            # 简单策略：如果caption就是这个词，直接返回英文
            if caption.strip() == chinese:
                return english
            # 否则尝试用AI翻译（如果需要更复杂的翻译）
    
    # 如果简单映射失败，使用AI翻译
    try:
        api_key = os.getenv("CHAT_API_KEY", os.getenv("SEARCH_API_KEY", ""))
        api_base = os.getenv("CHAT_BASE_URL", os.getenv("SEARCH_BASE_URL", "https://yunwu.ai/v1"))
        model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        
        if api_key and api_base:
            prompt = f"Translate this Chinese caption to English (2-6 words, for CLIP model): {caption}\n\nOutput ONLY the English translation, no other text:"
            
            resp = requests.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 20
                },
                timeout=10
            )
            
            if resp.status_code == 200:
                translated = resp.json()["choices"][0]["message"]["content"].strip()
                # 清理可能的格式
                translated = translated.replace("Translation:", "").replace("翻译：", "").strip()
                if translated:
                    return translated
    except Exception as e:
        # 翻译失败，返回原caption（虽然可能影响CLIP分数）
        print(f"  ⚠️  Caption translation failed: {e}, using original caption")
    
    # 如果所有翻译都失败，返回原caption（虽然可能影响CLIP分数）
    return caption


def compute_group_score(pair_scores: List[float]) -> Dict[str, float]:
    """
    Compute various GroupScore aggregations.

    Args:
        pair_scores: List of individual pair similarity scores

    Returns:
        Dictionary with different aggregation methods
    """
    if not pair_scores:
        return {
            "mean": 0.0,
            "harmonic": 0.0,
            "min": 0.0,
            "geometric": 0.0
        }

    scores = np.array(pair_scores)

    # Mean (arithmetic average)
    mean_score = np.mean(scores)

    # Harmonic mean (penalizes low scores more)
    if np.all(scores > 0):
        harmonic_score = len(scores) / np.sum(1.0 / scores)
    else:
        harmonic_score = 0.0

    # Minimum (worst case)
    min_score = np.min(scores)

    # Geometric mean (balanced aggregation)
    if np.all(scores > 0):
        geometric_score = np.exp(np.mean(np.log(scores)))
    else:
        geometric_score = 0.0

    return {
        "mean": float(mean_score),
        "harmonic": float(harmonic_score),
        "min": float(min_score),
        "geometric": float(geometric_score)
    }


def evaluate_sequence_layout(sequence: List[Dict]) -> Dict:
    """
    Evaluate the layout quality of content sequence.
    
    Checks:
    - Consecutive same-type elements (too many texts/images in a row)
    - Link placement (shouldn't be at the beginning)
    - Text paragraph length distribution
    
    Args:
        sequence: List of content elements from extract_content_sequence()
    
    Returns:
        Dictionary with layout score and issues
    """
    if not sequence:
        return {
            "score": 1.0,
            "issues": [],
            "stats": {"total_elements": 0}
        }
    
    issues = []
    
    # 1. Check consecutive same-type elements
    consecutive_count = 1
    prev_type = None
    
    for item in sequence:
        if item['type'] == prev_type:
            consecutive_count += 1
            if item['type'] == 'text' and consecutive_count > 3:
                issues.append(f"连续{consecutive_count}段文字，建议插入图片或链接")
            elif item['type'] == 'image' and consecutive_count > 2:
                issues.append(f"连续{consecutive_count}张图片，建议穿插文字说明")
        else:
            consecutive_count = 1
            prev_type = item['type']
    
    # 2. Check if link is at the beginning (not recommended)
    if sequence and sequence[0]['type'] == 'link':
        issues.append("开头不建议直接放链接，应该先有引入文字")
    
    # 3. Check text paragraph length distribution
    text_items = [s for s in sequence if s['type'] == 'text']
    if text_items:
        lengths = [s['char_count'] for s in text_items]
        avg_length = sum(lengths) / len(lengths)
        
        for s in text_items:
            if s['char_count'] > avg_length * 2 and s['char_count'] > 300:
                issues.append(f"第{s['index']+1}段文字过长({s['char_count']}字)，建议拆分")
    
    # 4. Check overall balance
    text_count = len([s for s in sequence if s['type'] == 'text'])
    image_count = len([s for s in sequence if s['type'] == 'image'])
    link_count = len([s for s in sequence if s['type'] == 'link'])
    
    # Text-only post should not be penalized
    if text_count > 0 and image_count == 0:
        pass  # Text-only is OK
    elif image_count > 0 and text_count == 0:
        issues.append("只有图片没有文字说明")
    
    # Calculate score: 1.0 - (0.1 * num_issues), min 0.0
    base_score = 1.0
    penalty_per_issue = 0.1
    layout_score = max(0.0, base_score - len(issues) * penalty_per_issue)
    
    return {
        "score": layout_score,
        "issues": issues,
        "stats": {
            "total_elements": len(sequence),
            "text_count": text_count,
            "image_count": image_count,
            "link_count": link_count
        }
    }


def evaluate_file_links(
    html_path: str,
    evaluator: TextEmbeddingEvaluator,
    verbose: bool = True
) -> GroupScoreResult:
    """
    Evaluate a single HTML file for link-text relevance (Hupu posts).
    
    Args:
        html_path: Path to the HTML file
        evaluator: TextEmbeddingEvaluator instance
        verbose: Whether to print detailed output
        
    Returns:
        GroupScoreResult object
    """
    parser = HTMLParser(html_path)
    pairs = parser.extract_link_text_pairs()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating Link-Text Relevance: {Path(html_path).name}")
        print(f"Found {len(pairs)} link-text pairs")
        
        if len(pairs) == 0:
            print(f"⚠️  No links found in this post")
            print(f"    GroupScore will be N/A for posts without links")
        
        print(f"{'='*60}")
    
    pair_scores_list = []
    combined_scores = []
    
    # Prepare all text pairs for batch processing
    text_pairs = []
    for pair in pairs:
        text_pairs.append((pair.link_title, pair.preceding_text))
        text_pairs.append((pair.link_title, pair.following_text))
        text_pairs.append((pair.link_title, pair.combined_text))
    
    # Compute all similarities in parallel (batch)
    all_similarities = evaluator.compute_similarities_batch(text_pairs, max_workers=5)
    
    # Process results
    for i, pair in enumerate(pairs):
        if verbose:
            print(f"\n[Link {i+1}] {pair.link_title}")
            print(f"  Platform: {pair.link_platform}")
            print(f"  Preceding text: {pair.preceding_text[:100]}..." if len(pair.preceding_text) > 100 else f"  Preceding text: {pair.preceding_text}")
            print(f"  Following text: {pair.following_text[:100]}..." if len(pair.following_text) > 100 else f"  Following text: {pair.following_text}")
        
        # Extract scores from batch results
        idx = i * 3
        preceding_score = all_similarities[idx]
        following_score = all_similarities[idx + 1]
        combined_score = all_similarities[idx + 2]
        
        link_score = LinkScore(
            link_title=pair.link_title,
            link_platform=pair.link_platform,
            preceding_score=preceding_score,
            following_score=following_score,
            combined_score=combined_score
        )
        pair_scores_list.append(asdict(link_score))
        combined_scores.append(combined_score)
        
        if verbose:
            print(f"  Scores:")
            print(f"    - Preceding text:  {preceding_score:.4f}")
            print(f"    - Following text:  {following_score:.4f}")
            print(f"    - Combined text:   {combined_score:.4f}")
            print(f"    → Final score: {combined_score:.4f}")
    
    # Compute GroupScore aggregations
    group_scores = compute_group_score(combined_scores)
    
    result = GroupScoreResult(
        file_name=Path(html_path).name,
        file_path=str(html_path),  # Include full path for thread-safety verification
        evaluation_type="link-text",
        num_pairs=len(pairs),
        pair_scores=pair_scores_list,
        group_score_mean=group_scores["mean"],
        group_score_harmonic=group_scores["harmonic"],
        group_score_min=group_scores["min"],
        group_score_geometric=group_scores["geometric"]
    )
    
    # Evaluate content sequence layout
    sequence = parser.extract_content_sequence()
    layout_eval = evaluate_sequence_layout(sequence)
    
    if verbose:
        print(f"\n{'-'*60}")
        if len(pairs) > 0:
            print("GroupScore Summary (Link-Text Relevance):")
            print(f"  - Mean:      {result.group_score_mean:.4f}")
            print(f"  - Harmonic:  {result.group_score_harmonic:.4f}")
            print(f"  - Geometric: {result.group_score_geometric:.4f}")
            print(f"  - Min:       {result.group_score_min:.4f}")
        else:
            print("GroupScore Summary: N/A (no links)")
        
        # Print layout evaluation
        print(f"\n{'-'*60}")
        print(f"Layout Quality Score: {layout_eval['score']:.4f}")
        print(f"Content Stats:")
        print(f"  - Total elements: {layout_eval['stats']['total_elements']}")
        print(f"  - Text blocks:    {layout_eval['stats']['text_count']}")
        print(f"  - Images:         {layout_eval['stats']['image_count']}")
        print(f"  - Links:          {layout_eval['stats']['link_count']}")
        
        if layout_eval['issues']:
            print(f"\nLayout Issues Found:")
            for issue in layout_eval['issues']:
                print(f"  ⚠️  {issue}")
        else:
            print(f"\n✅ No layout issues found")
    
    return result


def evaluate_file(
    html_path: str,
    evaluator: CLIPEvaluator,
    use_combined: bool = True,
    verbose: bool = True
) -> GroupScoreResult:
    """
    Evaluate a single HTML file for image-text consistency.
    
    Thread Safety:
    - Each thread processes a DIFFERENT html_path (ensured by process_user)
    - Each thread creates its own HTMLParser instance (independent state)
    - Each thread creates its own result lists (pair_scores_list, caption_scores)
    - The shared CLIPEvaluator is thread-safe for inference (read-only model)
    - Each thread's computation is completely independent
    - Results include html_path to ensure correct product-score mapping
    
    The score returned is guaranteed to correspond to the html_path provided,
    even when multiple threads are running concurrently.

    Args:
        html_path: Path to the HTML file (unique per thread)
        evaluator: CLIPEvaluator instance (shared, but thread-safe)
        use_combined: Whether to use combined text for scoring
        verbose: Whether to print detailed output

    Returns:
        GroupScoreResult object (contains html_path for verification)
    """
    # Thread-safe: Each thread creates its own parser instance
    parser = HTMLParser(html_path)
    pairs = parser.extract_image_text_pairs()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {Path(html_path).name}")
        print(f"Found {len(pairs)} image-text pairs")
        
        # 如果没有图片，给出提示
        if len(pairs) == 0:
            print(f"⚠️  No images found in this post (text-only post)")
            print(f"    GroupScore will be N/A for text-only posts")
        
        print(f"{'='*60}")

    pair_scores_list = []
    caption_scores = []

    for i, pair in enumerate(pairs):
        if verbose:
            print(f"\n[Image {i+1}] {pair.image_name}")
            if pair.caption:
                print(f"  Caption: {pair.caption}")
            else:
                print(f"  ⚠️  No caption found")

        # 🎯 核心修改：只使用caption计算GroupScore
        # CLIP模型基于简单的图片-文本对训练，不能用于计算图片和整段上下文的匹配度
        # 只使用简洁的caption（关键词）来计算相似度
        # ⚠️ CLIP模型在英文上表现更好，所以caption应该是英文
        caption_score = None
        if pair.caption:
            # 清理caption：移除"图X:"前缀（如果有）
            clean_caption = pair.caption
            if clean_caption.startswith("图") and ":" in clean_caption:
                clean_caption = clean_caption.split(":", 1)[1].strip()
            elif clean_caption.startswith("Image") and ":" in clean_caption:
                clean_caption = clean_caption.split(":", 1)[1].strip()
            
            # 确保caption是英文（CLIP在英文上表现更好）
            # 如果从data-caption-en读取，应该已经是英文
            # 但如果降级到文本内容读取，可能是中文，需要翻译
            english_caption = _ensure_english_caption(clean_caption)
            
            if verbose and clean_caption != english_caption:
                print(f"  ℹ️  Translated caption: '{clean_caption}' -> '{english_caption}'")
            
            caption_score = evaluator.compute_similarity(pair.image_path, english_caption)
            caption_scores.append(caption_score)
        else:
            # 如果没有caption，分数为0
            caption_score = 0.0
            caption_scores.append(caption_score)
            if verbose:
                print(f"  ⚠️  Warning: No caption available, score set to 0.0")

        # 保留其他分数用于调试（但不用于GroupScore计算）
        preceding_score = evaluator.compute_similarity(pair.image_path, pair.preceding_text) if pair.preceding_text else 0.0
        following_score = evaluator.compute_similarity(pair.image_path, pair.following_text) if pair.following_text else 0.0
        combined_score = evaluator.compute_similarity(pair.image_path, pair.combined_text) if pair.combined_text else 0.0

        pair_score = PairScore(
            image_name=pair.image_name,
            preceding_score=preceding_score,
            following_score=following_score,
            combined_score=combined_score,
            caption_score=caption_score
        )
        pair_scores_list.append(asdict(pair_score))

        if verbose:
            print(f"  Scores:")
            print(f"    - Caption (used for GroupScore): {caption_score:.4f}")
            if verbose and (preceding_score > 0 or following_score > 0):
                print(f"    - Preceding text (debug):  {preceding_score:.4f}")
                print(f"    - Following text (debug):  {following_score:.4f}")
                print(f"    - Combined text (debug):   {combined_score:.4f}")
            print(f"    → Final score: {caption_score:.4f}")

    # Compute GroupScore aggregations (只使用caption分数)
    group_scores = compute_group_score(caption_scores)

    # Create result with html_path included for thread-safety verification
    result = GroupScoreResult(
        file_name=Path(html_path).name,
        file_path=str(html_path),  # Include full path for verification
        evaluation_type="image-text",
        num_pairs=len(pairs),
        pair_scores=pair_scores_list,
        group_score_mean=group_scores["mean"],
        group_score_harmonic=group_scores["harmonic"],
        group_score_min=group_scores["min"],
        group_score_geometric=group_scores["geometric"]
    )

    # Evaluate content sequence layout
    sequence = parser.extract_content_sequence()
    layout_eval = evaluate_sequence_layout(sequence)

    if verbose:
        print(f"\n{'-'*60}")
        if len(pairs) > 0:
            print("GroupScore Summary (Image-Text Consistency):")
            print(f"  - Mean:      {result.group_score_mean:.4f}")
            print(f"  - Harmonic:  {result.group_score_harmonic:.4f}")
            print(f"  - Geometric: {result.group_score_geometric:.4f}")
            print(f"  - Min:       {result.group_score_min:.4f}")
        else:
            print("GroupScore Summary: N/A (no images)")
        
        # Print layout evaluation
        print(f"\n{'-'*60}")
        print(f"Layout Quality Score: {layout_eval['score']:.4f}")
        print(f"Content Stats:")
        print(f"  - Total elements: {layout_eval['stats']['total_elements']}")
        print(f"  - Text blocks:    {layout_eval['stats']['text_count']}")
        print(f"  - Images:         {layout_eval['stats']['image_count']}")
        print(f"  - Links:          {layout_eval['stats']['link_count']}")
        
        if layout_eval['issues']:
            print(f"\nLayout Issues Found:")
            for issue in layout_eval['issues']:
                print(f"  ⚠️  {issue}")
        else:
            print(f"\n✅ No layout issues found")

    return result


def evaluate_file_auto(
    html_path: str,
    clip_evaluator: Optional[CLIPEvaluator] = None,
    text_evaluator: Optional[TextEmbeddingEvaluator] = None,
    verbose: bool = True
) -> GroupScoreResult:
    """
    Automatically detect content type and evaluate accordingly.
    
    Args:
        html_path: Path to the HTML file
        clip_evaluator: CLIPEvaluator instance (for image-text)
        text_evaluator: TextEmbeddingEvaluator instance (for link-text)
        verbose: Whether to print detailed output
        
    Returns:
        GroupScoreResult object
    """
    parser = HTMLParser(html_path)
    
    # Check what content types are present
    image_pairs = parser.extract_image_text_pairs()
    link_pairs = parser.extract_link_text_pairs()
    
    num_images = len(image_pairs)
    num_links = len(link_pairs)
    
    if verbose:
        print(f"\n🔍 自动检测内容类型: {Path(html_path).name}")
        print(f"   - 图片数量: {num_images}")
        print(f"   - 链接数量: {num_links}")
    
    # Decide evaluation type based on content
    if num_images > 0:
        # Prioritize image-text evaluation if images are present
        if verbose:
            print(f"   → 使用 Image-Text 评估模式 (CLIP)")
        if not clip_evaluator:
            raise ValueError("CLIPEvaluator is required for image-text evaluation")
        return evaluate_file(html_path, clip_evaluator, verbose=verbose)
    
    elif num_links > 0:
        # Use link-text evaluation if only links are present
        if verbose:
            print(f"   → 使用 Link-Text 评估模式 (Text Embedding)")
        if not text_evaluator:
            raise ValueError("TextEmbeddingEvaluator is required for link-text evaluation")
        return evaluate_file_links(html_path, text_evaluator, verbose=verbose)
    
    else:
        # No images or links, return empty result
        if verbose:
            print(f"   ⚠️  无图片或链接，跳过评估")
        return GroupScoreResult(
            file_name=Path(html_path).name,
            file_path=str(html_path),  # Include full path for thread-safety verification
            evaluation_type="none",
            num_pairs=0,
            pair_scores=[],
            group_score_mean=0.0,
            group_score_harmonic=0.0,
            group_score_min=0.0,
            group_score_geometric=0.0
        )


def evaluate_folder(
    folder_path: str,
    output_file: Optional[str] = None,
    model_name: str = "ViT-B/32",
    verbose: bool = True
) -> List[GroupScoreResult]:
    """
    Evaluate all HTML files in a folder.

    Args:
        folder_path: Path to the folder containing HTML files
        output_file: Path to save JSON results (optional)
        model_name: CLIP model variant to use
        verbose: Whether to print detailed output

    Returns:
        List of GroupScoreResult objects
    """
    folder = Path(folder_path)
    html_files = list(folder.glob("*.html"))

    if not html_files:
        print(f"No HTML files found in {folder_path}")
        return []

    print(f"Found {len(html_files)} HTML files to evaluate")

    # Initialize CLIP evaluator
    evaluator = CLIPEvaluator(model_name=model_name)

    results = []
    for html_file in html_files:
        result = evaluate_file(str(html_file), evaluator, verbose=verbose)
        results.append(result)

    # Print overall summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Overall Summary")
        print(f"{'='*60}")
        for result in results:
            print(f"\n{result.file_name}:")
            print(f"  GroupScore (Mean): {result.group_score_mean:.4f}")

    # Save results to JSON
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(r) for r in results],
                f,
                ensure_ascii=False,
                indent=2
            )
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate image-text consistency using GroupScore metric"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=".",
        help="Path to folder containing HTML files (default: current directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="groupscore_results.json",
        help="Output JSON file path (default: groupscore_results.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"],
        help="CLIP model variant to use (default: ViT-B/32)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    evaluate_folder(
        folder_path=args.folder,
        output_file=args.output,
        model_name=args.model,
        verbose=not args.quiet
    )


def test_generated_posts(
    commentproduct_path: Optional[str] = None,
    itproduct_path: Optional[str] = None,
    model_name: str = "ViT-B/32",
    num_samples: int = 20
):
    """
    Test evaluation on generated posts.
    
    Args:
        commentproduct_path: Path to discussion_post.html (commentproduct)
        itproduct_path: Path to image_text.html (itproduct)
        model_name: CLIP model variant to use
        num_samples: Number of samples to test from each type (default: 5)
    """
    script_dir = Path(__file__).parent
    # Project root is parent of script_dir (if in tools/) or script_dir itself
    if script_dir.name == "tools":
        project_root = script_dir.parent
    else:
        project_root = script_dir
    
    print("\n" + "="*80)
    print("🧪 TESTING GROUPSCORE EVALUATION ON GENERATED POSTS")
    print("="*80)
    
    # Initialize evaluators (both CLIP and Text Embedding)
    print("\n📦 初始化评估器...")
    clip_evaluator = None
    text_evaluator = None
    
    try:
        clip_evaluator = CLIPEvaluator(model_name=model_name)
        print("   ✅ CLIP评估器已加载（用于图文评估）")
    except Exception as e:
        print(f"   ⚠️  CLIP评估器加载失败: {e}")
    
    try:
        text_evaluator = TextEmbeddingEvaluator()
        print("   ✅ 文本嵌入评估器已加载（用于链接评估）")
    except Exception as e:
        print(f"   ⚠️  文本嵌入评估器加载失败: {e}")
    
    if not clip_evaluator and not text_evaluator:
        print("❌ 没有可用的评估器！")
        return []
    
    results = []
    
    # Collect test files
    test_files = []
    
    # If specific paths provided, use them
    if commentproduct_path:
        test_files.append(("commentproduct", Path(commentproduct_path)))
    if itproduct_path:
        test_files.append(("itproduct", Path(itproduct_path)))
    
    # Otherwise, auto-discover samples
    if not test_files:
        print(f"\n🔍 Auto-discovering test samples (max {num_samples} per type)...")
        
        # Find CommentProduct samples
        commentproduct_dir = project_root / "generated_posts"
        if commentproduct_dir.exists():
            comment_samples = []
            for subdir in sorted(commentproduct_dir.iterdir()):
                if subdir.is_dir() and not subdir.name.startswith('old'):
                    html_file = subdir / "discussion_post.html"
                    if html_file.exists():
                        comment_samples.append(html_file)
            
            # Select samples (evenly distributed)
            if comment_samples:
                step = max(1, len(comment_samples) // num_samples)
                selected = comment_samples[::step][:num_samples]
                test_files.extend([("commentproduct", f) for f in selected])
                print(f"   ✅ Found {len(selected)} CommentProduct samples")
        
        # Find ITProduct samples
        itproduct_dir = project_root / "generated_it"
        if itproduct_dir.exists():
            it_samples = []
            for subdir in sorted(itproduct_dir.iterdir()):
                if subdir.is_dir():
                    html_file = subdir / "image_text.html"
                    if html_file.exists():
                        it_samples.append(html_file)
            
            # Select samples (evenly distributed)
            if it_samples:
                step = max(1, len(it_samples) // num_samples)
                selected = it_samples[::step][:num_samples]
                test_files.extend([("itproduct", f) for f in selected])
                print(f"   ✅ Found {len(selected)} ITProduct samples")
    
    if not test_files:
        print("❌ No test files found!")
        return []
    
    print(f"\n📊 Testing {len(test_files)} files total...\n")
    
    # Evaluate each file
    for idx, (post_type, file_path) in enumerate(test_files, 1):
        if not file_path.exists():
            print(f"\n⚠️  File not found: {file_path}")
            continue
        
        type_label = "CommentProduct (虎扑)" if post_type == "commentproduct" else "ITProduct (小红书)"
        print(f"\n{'='*80}")
        print(f"📝 Test {idx}/{len(test_files)}: {type_label}")
        print(f"   File: {file_path.parent.name}/{file_path.name}")
        print(f"{'='*80}")
        
        try:
            result = evaluate_file_auto(
                str(file_path), 
                clip_evaluator=clip_evaluator,
                text_evaluator=text_evaluator,
                verbose=True
            )
            results.append((post_type, file_path, result))
        except Exception as e:
            print(f"   ❌ 评估失败: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 OVERALL SUMMARY")
    print("="*80)
    
    # Group by type
    comment_results = [(f, r) for t, f, r in results if t == "commentproduct"]
    it_results = [(f, r) for t, f, r in results if t == "itproduct"]
    
    def print_type_summary(type_name, type_results):
        if not type_results:
            return
        
        print(f"\n{type_name}:")
        print(f"  Tested: {len(type_results)} files")
        
        # Detect evaluation type
        eval_types = [r.evaluation_type for f, r in type_results if r.num_pairs > 0]
        primary_eval_type = eval_types[0] if eval_types else "none"
        
        # Statistics for posts with content
        scores = [r.group_score_mean for f, r in type_results if r.num_pairs > 0]
        
        if scores:
            content_label = "图片" if primary_eval_type == "image-text" else "链接" if primary_eval_type == "link-text" else "元素"
            print(f"  With {content_label}: {len(scores)} files")
            print(f"    Avg GroupScore: {sum(scores)/len(scores):.4f}")
            print(f"    Max GroupScore: {max(scores):.4f}")
            print(f"    Min GroupScore: {min(scores):.4f}")
            
            # Count by quality
            good = sum(1 for s in scores if s >= 0.65)
            medium = sum(1 for s in scores if 0.50 <= s < 0.65)
            poor = sum(1 for s in scores if s < 0.50)
            
            print(f"    Quality Distribution:")
            print(f"      ✅ Good (≥0.65):     {good}")
            print(f"      ⚠️  Medium (0.50-0.65): {medium}")
            print(f"      ❌ Poor (<0.50):     {poor}")
        
        no_content = len(type_results) - len(scores)
        if no_content > 0:
            print(f"  No evaluable content: {no_content} files")
        
        # Show individual scores
        print(f"\n  Individual Results:")
        for file_path, result in type_results:
            folder_name = file_path.parent.name
            if result.num_pairs > 0:
                score = result.group_score_mean
                eval_label = "图片" if result.evaluation_type == "image-text" else "链接" if result.evaluation_type == "link-text" else "元素"
                if score >= 0.65:
                    status = "✅"
                elif score >= 0.50:
                    status = "⚠️ "
                else:
                    status = "❌"
                print(f"    {status} {folder_name}: {score:.4f} ({result.num_pairs} {eval_label})")
            else:
                print(f"    ℹ️  {folder_name}: N/A (no content)")
    
    print_type_summary("CommentProduct (虎扑讨论帖)", comment_results)
    print_type_summary("ITProduct (小红书图文帖)", it_results)
    
    print("\n" + "="*80)
    print("✅ Testing Complete!")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    # Determine project root
    script_dir = Path(__file__).parent
    if script_dir.name == "tools":
        project_root = script_dir.parent
    else:
        project_root = script_dir

    # Check if running as script or module
    import sys
    if len(sys.argv) == 1:
        # Default: Test generated posts with 5 samples each
        print("🧪 Running test evaluation on generated posts...")
        test_generated_posts(num_samples=5)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Explicit test mode
        # Usage: python evaluate_groupscore.py --test [num_samples]
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        print(f"🧪 Running test with {num_samples} samples per type...")
        test_generated_posts(num_samples=num_samples)
    elif len(sys.argv) > 1 and sys.argv[1] == "--file":
        # Test specific files
        # Usage: python evaluate_groupscore.py --file <path1> [path2]
        commentproduct = sys.argv[2] if len(sys.argv) > 2 else None
        itproduct = sys.argv[3] if len(sys.argv) > 3 else None
        test_generated_posts(commentproduct_path=commentproduct, itproduct_path=itproduct, num_samples=0)
    else:
        main()

