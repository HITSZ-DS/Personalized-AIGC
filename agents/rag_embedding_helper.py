"""
RAG Embedding Helper for retrieving relevant examples based on user preferences
Supports multiple datasets and efficient caching

将 ProductGenerator 中的固定样例检索方式改为 **RAG（Retrieval-Augmented Generation）** 模式：
- 根据用户的 **top1 preference** 
- 从该用户的 **historical** 和 **recommended** 文件夹
- 使用 **embedding 向量相似度检索** 最相关的样例文件
- 支持 **多数据集**、**智能缓存** 和 **自动降级**
"""



import json
import os
import pickle
import time
from openai import OpenAI
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import glob
import hashlib


class RAGEmbeddingHelper:
    def __init__(self, api_key=None, api_base=None, cache_dir="embeddings_cache"):
        """
        Initialize RAG Embedding Helper
        
        Args:
            api_key: OpenAI API key
            api_base: API base URL
            cache_dir: Directory to store embedding caches
        """
        self.api_key = api_key or os.getenv("CHAT_API_KEY")
        self.api_base = api_base or os.getenv("CHAT_BASE_URL")
        
        # Initialize OpenAI client with new SDK (v1.0.0+)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
            
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.embedding_model = "text-embedding-3-small"
        self.save_lock = Lock()
        
    def get_cache_filename(self, dataset_name, user_id):
        """
        Generate cache filename for a specific dataset and user
        
        Args:
            dataset_name: Name of dataset (e.g., 'hupu', 'redbook')
            user_id: User ID
            
        Returns:
            Cache file path
        """
        filename = f"{dataset_name}_user_{user_id}_embeddings.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def load_embeddings_cache(self, cache_file):
        """Load embeddings from cache file"""
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            return cache_data
        except FileNotFoundError:
            return {"embeddings": {}, "metadata": {}}
        except Exception as e:
            print(f"⚠️ Could not load cache {cache_file}: {e}")
            return {"embeddings": {}, "metadata": {}}
    
    def save_embeddings_cache(self, cache_data, cache_file):
        """Save embeddings to cache file"""
        try:
            with self.save_lock:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
        except Exception as e:
            print(f"⚠️ Error saving cache {cache_file}: {e}")
    
    def get_text_hash(self, text):
        """Generate hash for text content"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text, max_retries=3):
        """
        Get embedding for a single text with retry mechanism
        
        Args:
            text: Text to embed
            max_retries: Maximum number of retries
            
        Returns:
            Embedding vector as numpy array
        """
        for attempt in range(max_retries):
            try:
                # Use new OpenAI SDK (v1.0.0+) syntax
                response = self.client.embeddings.create(
                    input=text[:8000],  # Limit text length to avoid API errors
                    model=self.embedding_model
                )
                embedding = response.data[0].embedding
                return np.array(embedding)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 1
                    print(f"   Retry {attempt + 1}/{max_retries} for embedding: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Failed after {max_retries} attempts: {e}")
                    raise e
    
    def collect_user_files(self, dataset_root, user_id, folders=["historical", "recommended"]):
        """
        Collect all .txt files from user's historical and recommended folders
        
        Args:
            dataset_root: Root directory of dataset (e.g., 'download/hupu')
            user_id: User ID
            folders: List of folder names to search
            
        Returns:
            List of dicts with file info: {"path": str, "content": str, "folder": str, "post_id": str}
        """
        user_dir = os.path.join(dataset_root, str(user_id))
        
        if not os.path.exists(user_dir):
            print(f"⚠️ User directory not found: {user_dir}")
            return []
        
        files = []
        for folder in folders:
            folder_path = os.path.join(user_dir, folder)
            if not os.path.exists(folder_path):
                continue
            
            # Find all .txt files in subdirectories
            txt_files = glob.glob(os.path.join(folder_path, "*", "*.txt"))
            
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if not content or len(content) < 50:  # Skip empty or very short files
                        continue
                    
                    # Extract post_id from path
                    post_id = os.path.basename(os.path.dirname(txt_file))
                    
                    files.append({
                        "path": txt_file,
                        "content": content,
                        "folder": folder,
                        "post_id": post_id,
                        "filename": os.path.basename(txt_file)
                    })
                except Exception as e:
                    print(f"   ⚠️ Error reading {txt_file}: {e}")
        
        return files
    
    def build_embeddings_for_user(self, dataset_name, dataset_root, user_id, 
                                   max_workers=10, use_cache=True):
        """
        Build embeddings for all files of a specific user
        
        Args:
            dataset_name: Name of dataset (e.g., 'hupu')
            dataset_root: Root directory of dataset
            user_id: User ID
            max_workers: Number of parallel workers
            use_cache: Whether to use cached embeddings
            
        Returns:
            Dict with embeddings and metadata
        """
        cache_file = self.get_cache_filename(dataset_name, user_id)
        
        # Load existing cache if enabled
        if use_cache:
            cache_data = self.load_embeddings_cache(cache_file)
            print(f"📦 Loaded {len(cache_data['embeddings'])} cached embeddings")
        else:
            cache_data = {"embeddings": {}, "metadata": {}}
        
        # Collect all user files
        files = self.collect_user_files(dataset_root, user_id)
        
        if not files:
            print(f"⚠️ No files found for user {user_id}")
            return cache_data
        
        print(f"📁 Found {len(files)} files for user {user_id}")
        
        # Filter files that need embedding
        files_to_process = []
        for file_info in files:
            file_path = file_info["path"]
            text_hash = self.get_text_hash(file_info["content"])
            
            # Check if already cached
            if file_path in cache_data["embeddings"]:
                cached_hash = cache_data["metadata"].get(file_path, {}).get("text_hash")
                if cached_hash == text_hash:
                    continue  # Skip if content hasn't changed
            
            files_to_process.append(file_info)
        
        if not files_to_process:
            print(f"✅ All files already embedded (cache hit)")
            return cache_data
        
        print(f"🔄 Processing {len(files_to_process)} new/updated files...")
        
        # Process embeddings in parallel
        def process_single_file(file_info):
            try:
                embedding = self.get_embedding(file_info["content"])
                time.sleep(0.05)  # Rate limiting
                return file_info, embedding, None
            except Exception as e:
                return file_info, None, e
        
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, f) for f in files_to_process]
            
            for future in tqdm(as_completed(futures), total=len(files_to_process), 
                              desc="Generating embeddings"):
                file_info, embedding, error = future.result()
                
                if error is None:
                    file_path = file_info["path"]
                    text_hash = self.get_text_hash(file_info["content"])
                    
                    cache_data["embeddings"][file_path] = embedding
                    cache_data["metadata"][file_path] = {
                        "text_hash": text_hash,
                        "folder": file_info["folder"],
                        "post_id": file_info["post_id"],
                        "filename": file_info["filename"],
                        "content_preview": file_info["content"][:200]
                    }
                    
                    completed += 1
                    
                    # Save cache every 20 files
                    if completed % 20 == 0:
                        self.save_embeddings_cache(cache_data, cache_file)
                else:
                    print(f"   ❌ Failed: {file_info['path']}")
        
        # Final save
        self.save_embeddings_cache(cache_data, cache_file)
        print(f"💾 Saved {len(cache_data['embeddings'])} embeddings to cache")
        
        return cache_data
    
    def clear_user_cache(self, dataset_name, user_id):
        """
        Clear cached embeddings for a specific user (for cleanup after processing)
        
        Args:
            dataset_name: Name of dataset (e.g., 'hupu', 'redbook')
            user_id: User ID
        """
        cache_file = self.get_cache_filename(dataset_name, user_id)
        
        try:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🗑️  Cleared embedding cache for {dataset_name} user {user_id}")
                return True
            else:
                print(f"ℹ️  No cache file to clear for {dataset_name} user {user_id}")
                return False
        except Exception as e:
            print(f"⚠️  Failed to clear cache {cache_file}: {e}")
            return False
    
    def retrieve_top_k_similar(self, query_text, embeddings_data, top_k=5):
        """
        Retrieve top-k most similar files based on cosine similarity
        
        Args:
            query_text: Query text (e.g., user's top1 preference)
            embeddings_data: Dict with embeddings and metadata
            top_k: Number of results to return
            
        Returns:
            List of dicts with file info and similarity scores
        """
        if not embeddings_data["embeddings"]:
            print("⚠️ No embeddings available")
            return []
        
        # Get query embedding
        try:
            query_embedding = self.get_embedding(query_text)
        except Exception as e:
            print(f"❌ Failed to get query embedding: {e}")
            return []
        
        # Calculate cosine similarities
        similarities = []
        for file_path, file_embedding in embeddings_data["embeddings"].items():
            # Cosine similarity
            sim = np.dot(query_embedding, file_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(file_embedding)
            )
            
            metadata = embeddings_data["metadata"].get(file_path, {})
            similarities.append({
                "path": file_path,
                "similarity": float(sim),
                "folder": metadata.get("folder", "unknown"),
                "post_id": metadata.get("post_id", "unknown"),
                "filename": metadata.get("filename", "unknown"),
                "content_preview": metadata.get("content_preview", "")
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:top_k]
    
    def get_file_content(self, file_path):
        """Load full content of a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
            return ""


# Example usage
if __name__ == "__main__":
    helper = RAGEmbeddingHelper()
    
    # Example: Build embeddings for a user
    dataset_name = "hupu"
    dataset_root = "../download/hupu"
    user_id = "132349263326575"
    
    embeddings_data = helper.build_embeddings_for_user(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        user_id=user_id,
        max_workers=10,
        use_cache=True
    )
    
    # Example: Retrieve similar files
    query = "basketball game analysis and player performance discussion"
    results = helper.retrieve_top_k_similar(query, embeddings_data, top_k=5)
    
    print("\n🔍 Top 5 similar files:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity: {result['similarity']:.3f}")
        print(f"   Path: {result['path']}")
        print(f"   Preview: {result['content_preview'][:100]}...")
        print()

