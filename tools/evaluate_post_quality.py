"""
使用方法

### 1. 批量评价所有产品（推荐）

# 评价所有Redbook产品
python tools/evaluate_post_quality.py generated_redbook_it --batch-all

# 评价所有Hupu产品
python tools/evaluate_post_quality.py generated_hupu_posts --batch-all

# 自定义并发参数
python tools/evaluate_post_quality.py generated_redbook_it --batch-all --max-workers 4 --max-products 2
```

### 2. 评价单个产品目录

```bash
# 评价单个产品的所有版本
python tools/evaluate_post_quality.py generated_redbook_it/0_5435e123d6e4a965e190095a

# 评价单个Hupu产品
python tools/evaluate_post_quality.py generated_hupu_posts/2_100274844777875
```

### 3. 评价单个文件

```bash
# 评价单个HTML文件
python tools/evaluate_post_quality.py generated_redbook_it/0_5435e123d6e4a965e190095a/image_text_v0.html
"""

import os
import json
import sys
import re
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.evaluate_html_post import PostEvaluationPackage, evaluate_html_post


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension"""
    dimension: str
    score: float  # 0-10


@dataclass
class PostEvaluationResult:
    """Complete evaluation result for a post"""
    html_path: str
    overall_score: float  # Weighted average
    dimension_scores: List[DimensionScore]


class PostQualityEvaluator:
    """Evaluates post quality using AI"""
    
    # Dimension weights for Redbook (sum to 1.0) - 精简为4个维度
    DIMENSION_WEIGHTS_REDBOOK = {
        'content_quality': 0.30,      # 内容质量（逻辑+信息+语言）
        'platform_fit': 0.30,         # 平台适配性（风格+互动+吸引力）
        'human_likeness': 0.25,       # 拟人程度（真实+情感+自然）
        'visual_presentation': 0.15,  # 视觉呈现（图片+排版）
    }
    
    # Dimension weights for Hupu (sum to 1.0) - 精简为4个维度
    DIMENSION_WEIGHTS_HUPU = {
        'content_quality': 0.35,      # 内容质量（逻辑+观点+信息+语言）
        'link_relevance': 0.25,       # 链接相关性（重要特色）
        'platform_fit': 0.25,          # 平台适配性（风格+讨论引导+热点）
        'human_likeness': 0.15,       # 拟人程度（真实+自然）
    }
    
    # Evaluation dimensions for Redbook - 精简为4个维度
    DIMENSIONS_REDBOOK = {
        'content_quality': {
            'name': '内容质量',
            'description': '内容结构是否清晰，逻辑是否连贯，信息是否有价值，语言表达是否准确流畅'
        },
        'platform_fit': {
            'name': '平台适配性',
            'description': '是否符合小红书风格，标签使用是否恰当，是否吸引人，是否自然引导互动'
        },
        'human_likeness': {
            'name': '拟人程度',
            'description': '语言是否自然真实，是否有个人化表达，情感表达是否恰当，是否像真人写的'
        },
        'visual_presentation': {
            'name': '视觉呈现',
            'description': '图片与文字是否匹配，排版是否合理美观，视觉元素是否恰当'
        }
    }
    
    # Evaluation dimensions for Hupu - 精简为4个维度
    DIMENSIONS_HUPU = {
        'content_quality': {
            'name': '内容质量',
            'description': '内容结构是否清晰，逻辑论证是否严密，观点是否有深度，信息是否准确有价值，语言表达是否流畅'
        },
        'link_relevance': {
            'name': '链接相关性',
            'description': '链接与文本内容是否高度相关，链接是否支持论点，链接质量是否可靠'
        },
        'platform_fit': {
            'name': '平台适配性',
            'description': '是否符合虎扑论坛风格，语气是否恰当，是否能够引发讨论，是否紧跟热点话题'
        },
        'human_likeness': {
            'name': '拟人程度',
            'description': '语言是否自然，是否有个人化表达，内容是否真实可信，是否像真实用户发布'
        }
    }
    
    def __init__(self, use_vision_model: bool = True, post_type: str = 'redbook'):
        """
        Initialize evaluator
        
        Args:
            use_vision_model: Whether to use vision-capable model (can see images)
            post_type: 'redbook' or 'hupu'
        """
        self.use_vision_model = use_vision_model
        self.post_type = post_type.lower()
        
        # Select dimensions and weights based on post type
        if self.post_type == 'hupu':
            self.DIMENSIONS = self.DIMENSIONS_HUPU
            self.DIMENSION_WEIGHTS = self.DIMENSION_WEIGHTS_HUPU
        else:
            self.DIMENSIONS = self.DIMENSIONS_REDBOOK
            self.DIMENSION_WEIGHTS = self.DIMENSION_WEIGHTS_REDBOOK
        
        # Initialize OpenAI client with .env configuration (lines 9-12)
        self.client = OpenAI(
            api_key=os.getenv("CHAT_API_KEY"),
            base_url=os.getenv("CHAT_BASE_URL"),
        )
        self.model = os.getenv("CHAT_MODEL", "claude-sonnet-4-5-20250929")
    
    def _extract_version_info(self, html_path: str) -> Dict[str, any]:
        """Extract version information from HTML file path"""
        import re
        html_path_str = str(html_path)
        filename = Path(html_path).stem
        
        # Extract version number
        version_match = re.search(r'v(\d+)', filename)
        is_english = 'english' in filename.lower()
        
        if is_english:
            return {
                'version_type': 'english',
                'version_number': None,
                'is_early': False,
                'description': '这是经过AI翻译优化的最终版本'
            }
        elif version_match:
            version_num = int(version_match.group(1))
            return {
                'version_type': 'versioned',
                'version_number': version_num,
                'is_early': version_num == 0,  # 只有v0视为早期版本
                'description': f'这是第{version_num}轮迭代生成的版本'
            }
        else:
            return {
                'version_type': 'unknown',
                'version_number': None,
                'is_early': None,
                'description': '版本信息未知'
            }
    
    def create_evaluation_prompt(self, package: PostEvaluationPackage) -> str:
        """Create prompt for AI evaluation"""
        prompt_parts = []
        
        # Detect post type from package
        post_type = package.post_type
        platform_name = "小红书" if post_type == 'redbook' else "虎扑论坛"
        
        # Extract version information
        version_info = self._extract_version_info(package.html_path)
        
        prompt_parts.append(f"你是一个专业的社交媒体内容评价专家。请对以下AI生成的{platform_name}帖子进行全面评价。")
        prompt_parts.append("\n" + "=" * 60)
        prompt_parts.append("评价维度（每个维度0-10分）：")
        prompt_parts.append("=" * 60)
        
        for dim_id, dim_info in self.DIMENSIONS.items():
            prompt_parts.append(f"\n{dim_info['name']} ({dim_id}):")
            prompt_parts.append(f"  {dim_info['description']}")
        
        prompt_parts.append("\n" + "=" * 60)
        prompt_parts.append("帖子内容：")
        prompt_parts.append("=" * 60)
        
        # Add version information explicitly (subtle)
        if version_info['version_type'] != 'unknown':
            prompt_parts.append(f"\n版本信息：{version_info['description']}")
        
        # Add text content
        prompt_parts.append(package.to_prompt_format())
        
        # Add evaluation instructions
        prompt_parts.append("\n" + "=" * 60)
        prompt_parts.append("评价要求：")
        prompt_parts.append("=" * 60)
        
        # Subtle adjustment for early versions - more strict evaluation
        if version_info.get('is_early', False):
            evaluation_instructions = """
1. 对每个维度进行评分（0-10分，保留1位小数），请以高标准评价内容的完整性和质量
2. 特别关注内容是否存在需要改进的地方，评分时请严格把关
   - 对于逻辑性问题、视觉缺陷、语言不自然等问题，应相应扣分
   - 不要因为内容基本完整就给予高分，要关注细节和质量
   - 评分应反映内容的真实质量水平，避免过于宽松
3. 计算加权平均分作为总体得分

请以JSON格式输出结果，格式如下：
{
    "dimension_scores": [
        {
            "dimension": "content_quality",
            "score": 8.5
        },
        {
            "dimension": "platform_fit",
            "score": 8.0
        },
        ...
    ],
    "overall_score": 8.2
}

注意：只需要输出评分，不需要理由、建议或其他文字说明。
"""
        else:
            evaluation_instructions = """
1. 对每个维度进行评分（0-10分，保留1位小数）
2. 计算加权平均分作为总体得分

请以JSON格式输出结果，格式如下：
{
    "dimension_scores": [
        {
            "dimension": "content_quality",
            "score": 8.5
        },
        {
            "dimension": "platform_fit",
            "score": 8.0
        },
        ...
    ],
    "overall_score": 8.2
}

注意：只需要输出评分，不需要理由、建议或其他文字说明。
"""
        
        prompt_parts.append(evaluation_instructions)
        
        return "\n".join(prompt_parts)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def evaluate_with_ai(self, package: PostEvaluationPackage) -> PostEvaluationResult:
        """
        Evaluate post using AI
        
        Args:
            package: PostEvaluationPackage object
        
        Returns:
            PostEvaluationResult object
        """
        # Create prompt
        prompt = self.create_evaluation_prompt(package)
        
        # Prepare content for API
        messages = [{"role": "user", "content": prompt}]
        
        # Add screenshot if using vision model (only for redbook posts with images)
        # Only send screenshot, not individual images - screenshot contains all visual information
        if self.use_vision_model and self.post_type == 'redbook' and len(package.images) > 0:
            # Add screenshot if available
            if package.screenshot_path and package.screenshot_gen:
                screenshot_base64 = package.screenshot_gen.image_to_base64(package.screenshot_path)
                if screenshot_base64:
                    messages[0]["content"] = [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": screenshot_base64}
                        }
                    ]
        
        # Call AI API
        print("=" * 60)
        print("Calling AI API for evaluation...")
        print(f"Post Type: {self.post_type.upper()}")
        print(f"Model: {self.model}")
        print(f"Using vision: {self.use_vision_model and self.post_type == 'redbook' and len(package.images) > 0}")
        if isinstance(messages[0]["content"], list):
            has_screenshot = any(c.get('type') == 'image_url' for c in messages[0]['content'])
            if has_screenshot:
                print(f"Content: Text + Screenshot (contains all images and layout)")
            else:
                print(f"Content: Text only")
        print("=" * 60)
        
        response_text = None
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            response_text = completion.choices[0].message.content
            
            # Clean response (remove markdown code blocks if any)
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON response
            result_dict = json.loads(cleaned_response)
            
            # Convert to PostEvaluationResult
            dimension_scores = []
            for dim_data in result_dict.get("dimension_scores", []):
                dimension_scores.append(DimensionScore(
                    dimension=dim_data.get("dimension", ""),
                    score=float(dim_data.get("score", 0.0))
                ))
            
            # Calculate weighted score if not provided
            overall_score = result_dict.get("overall_score")
            if overall_score is None:
                overall_score = self.calculate_weighted_score(dimension_scores)
            
            result = PostEvaluationResult(
                html_path=str(package.html_path),
                overall_score=float(overall_score),
                dimension_scores=dimension_scores
            )
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
            if response_text:
                print(f"响应内容: {response_text[:500]}")
            raise
        except Exception as e:
            print(f"❌ API调用错误: {e}")
            if response_text:
                print(f"响应内容: {response_text[:500] if len(response_text) > 500 else response_text}")
            raise
    
    def calculate_weighted_score(self, dimension_scores: List[DimensionScore]) -> float:
        """Calculate weighted overall score using current dimension weights"""
        total_score = 0.0
        total_weight = 0.0
        
        for dim_score in dimension_scores:
            weight = self.DIMENSION_WEIGHTS.get(dim_score.dimension, 0.0)
            total_score += dim_score.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


def evaluate_single_post(html_path: str, 
                        generate_screenshot: bool = True,
                        use_vision: bool = True,
                        save_to_product_dir: bool = False,
                        post_type: Optional[str] = None) -> Dict:
    """
    Evaluate a single HTML post
    
    Args:
        html_path: Path to HTML file
        generate_screenshot: Whether to generate screenshot
        use_vision: Whether to use vision model
        save_to_product_dir: Whether to save individual evaluation JSON files (default False, only summary is saved)
        post_type: Post type ('redbook' or 'hupu'). If None, auto-detect from package
    
    Returns:
        Evaluation result dictionary
    """
    html_path_obj = Path(html_path)
    
    # Create evaluation package
    package = evaluate_html_post(html_path, generate_screenshot)
    
    # Determine post type
    detected_type = post_type or package.post_type
    
    # Create evaluator with detected post type
    evaluator = PostQualityEvaluator(use_vision_model=use_vision, post_type=detected_type)
    
    # Evaluate with AI
    print(f"\n{'='*60}")
    print(f"Evaluating: {html_path_obj.name}")
    print(f"{'='*60}\n")
    
    result = evaluator.evaluate_with_ai(package)
    
    # Convert to dict for saving
    result_dict = {
        'html_path': str(html_path_obj),
        'html_filename': html_path_obj.name,
        'post_type': detected_type,
        'overall_score': result.overall_score,
        'dimension_scores': [
            {
                'dimension': ds.dimension,
                'dimension_name': evaluator.DIMENSIONS.get(ds.dimension, {}).get('name', ds.dimension),
                'score': ds.score
            }
            for ds in result.dimension_scores
        ]
    }
    
    # Save to product directory only if explicitly requested
    # By default, individual files are not saved (only summary is saved)
    if save_to_product_dir:
        output_dir = html_path_obj.parent
        output_file = output_dir / f"{html_path_obj.stem}_evaluation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 评价结果已保存到: {output_file}")
    
    print(f"总体得分: {result.overall_score:.2f}/10.0")
    
    return result_dict


def find_all_post_versions(product_dir: Path) -> List[Path]:
    """
    Find all HTML post versions in a product directory
    
    Args:
        product_dir: Product directory path
    
    Returns:
        List of HTML file paths (v0, v1, v2, ..., english)
    """
    html_files = []
    
    # Find all version files (v0, v1, v2, etc.)
    version_files = sorted(product_dir.glob("*_v*.html"))
    html_files.extend(version_files)
    
    # Find english version
    english_files = list(product_dir.glob("*_english.html"))
    html_files.extend(english_files)
    
    # Also check for image_text_*.html (redbook) or discussion_post_*.html (hupu)
    if not html_files:
        html_files = list(product_dir.glob("*.html"))
        # Filter out evaluation packages and other non-post files
        html_files = [f for f in html_files if not f.name.endswith('_evaluation_package.html')]
    
    return sorted(html_files, key=lambda x: x.name)


def evaluate_product_versions(product_dir: Path,
                              generate_screenshot: bool = True,
                              use_vision: bool = True,
                              max_workers: int = 5,
                              shuffle_versions: bool = True,
                              skip_if_summary_exists: bool = True) -> Dict:
    """
    Evaluate all versions of a single product
    
    Args:
        product_dir: Product directory path
        generate_screenshot: Whether to generate screenshots
        use_vision: Whether to use vision model
        max_workers: Maximum concurrent workers
        shuffle_versions: Whether to shuffle version order to avoid bias
        skip_if_summary_exists: Whether to skip if summary file already exists (resume mechanism)
    
    Returns:
        Dictionary with evaluation results for all versions
    """
    # Check if summary already exists (resume mechanism)
    summary_file = product_dir / "post_evaluations_summary.json"
    if skip_if_summary_exists and summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                existing_summary = json.load(f)
            print(f"\n{'='*60}")
            print(f"产品: {product_dir.name}")
            print(f"⏭️  已存在评价汇总，跳过评价")
            print(f"   平均得分: {existing_summary.get('average_score', 0):.2f}/10.0")
            print(f"   已评价版本: {existing_summary.get('successful', 0)}/{existing_summary.get('total_versions', 0)}")
            print(f"{'='*60}")
            return existing_summary
        except Exception as e:
            print(f"⚠️  读取已有汇总文件失败: {e}，将重新评价")
    
    html_files = find_all_post_versions(product_dir)
    
    if not html_files:
        return {
            'product_dir': str(product_dir),
            'product_id': product_dir.name,
            'versions': [],
            'error': 'No HTML files found'
        }
    
    # Shuffle versions to avoid order bias (AI might favor first versions)
    if shuffle_versions:
        html_files = html_files.copy()
        random.shuffle(html_files)
        print(f"\n{'='*60}")
        print(f"产品: {product_dir.name}")
        print(f"找到 {len(html_files)} 个版本文件（已随机打乱顺序）")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"产品: {product_dir.name}")
        print(f"找到 {len(html_files)} 个版本文件")
        print(f"{'='*60}")
    
    results = {}
    errors = {}
    
    # Process files with limited concurrency to avoid API rate limits
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                evaluate_single_post,
                str(html_file),
                generate_screenshot=generate_screenshot,
                use_vision=use_vision,
                save_to_product_dir=False  # Don't save individual files, only summary
            ): html_file
            for html_file in html_files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            html_file = future_to_file[future]
            version_name = html_file.stem  # e.g., "image_text_v0" or "discussion_post_english"
            
            try:
                result = future.result()
                results[version_name] = result
                print(f"  ✅ {html_file.name}: {result.get('overall_score', 0):.2f}/10.0")
            except Exception as e:
                error_msg = str(e)
                errors[version_name] = error_msg
                print(f"  ❌ {html_file.name}: {error_msg}")
    
    # Compile summary
    summary = {
        'product_dir': str(product_dir),
        'product_id': product_dir.name,
        'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_versions': len(html_files),
        'successful': len(results),
        'failed': len(errors),
        'versions': {}
    }
    
    # Add version results
    for version_name, result in results.items():
        summary['versions'][version_name] = {
            'html_file': result.get('html_filename', ''),
            'post_type': result.get('post_type', 'unknown'),
            'overall_score': result.get('overall_score', 0.0),
            'dimension_scores': {
                ds['dimension']: {
                    'name': ds['dimension_name'],
                    'score': ds['score']
                }
                for ds in result.get('dimension_scores', [])
            }
        }
    
    # Add errors
    if errors:
        summary['errors'] = errors
    
    # Calculate average scores (exclude english version from statistics)
    # English version is evaluated but not included in average/max/min calculations
    if results:
        # Separate english and non-english versions
        non_english_results = {
            name: result for name, result in results.items() 
            if 'english' not in name.lower()
        }
        english_results = {
            name: result for name, result in results.items() 
            if 'english' in name.lower()
        }
        
        # Calculate statistics only from non-english versions
        if non_english_results:
            non_english_scores = [r.get('overall_score', 0) for r in non_english_results.values()]
            summary['average_score'] = sum(non_english_scores) / len(non_english_scores)
            summary['max_score'] = max(non_english_scores)
            summary['min_score'] = min(non_english_scores)
            
            # Find best version from non-english versions
            best_version = max(non_english_results.items(), key=lambda x: x[1].get('overall_score', 0))
            summary['best_version'] = {
                'name': best_version[0],
                'score': best_version[1].get('overall_score', 0)
            }
            summary['best_version_note'] = 'English version excluded from statistics and best version selection'
        else:
            # Fallback: if only english version exists
            summary['average_score'] = 0.0
            summary['max_score'] = 0.0
            summary['min_score'] = 0.0
            summary['best_version'] = {
                'name': 'N/A',
                'score': 0.0
            }
            summary['best_version_note'] = 'Only english version available, no statistics calculated'
        
        # Store english version score separately (if exists)
        if english_results:
            english_scores = [r.get('overall_score', 0) for r in english_results.values()]
            summary['english_version_score'] = english_scores[0] if len(english_scores) == 1 else sum(english_scores) / len(english_scores)
            summary['english_version_note'] = 'English version scored separately, not included in statistics'
    
    # Save summary to product directory
    summary_file = product_dir / "post_evaluations_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n  💾 评价汇总已保存到: {summary_file}")
    if results:
        print(f"  📊 平均得分: {summary['average_score']:.2f}/10.0 (English版本已排除)")
        best_name = summary['best_version']['name']
        best_score = summary['best_version']['score']
        if best_name != 'N/A':
            print(f"  🏆 最佳版本: {best_name} ({best_score:.2f}/10.0) [English版本已排除]")
        else:
            print(f"  🏆 最佳版本: N/A (仅有English版本)")
        
        # Show english version score if exists
        if 'english_version_score' in summary:
            print(f"  🌐 English版本得分: {summary['english_version_score']:.2f}/10.0 (单独评分，不参与统计)")
    
    return summary


def batch_evaluate_all_products(base_dir: str,
                                generate_screenshot: bool = True,
                                use_vision: bool = True,
                                max_workers_per_product: int = 5,
                                max_concurrent_products: int = 8,
                                skip_if_summary_exists: bool = True) -> List[Dict]:
    """
    Batch evaluate all products in a directory (e.g., generated_redbook_it or generated_hupu_posts)
    
    Args:
        base_dir: Base directory containing product directories
        generate_screenshot: Whether to generate screenshots
        use_vision: Whether to use vision model
        max_workers_per_product: Max concurrent versions per product
        max_concurrent_products: Max concurrent products (to avoid overwhelming API)
        skip_if_summary_exists: Whether to skip products that already have summary files (resume mechanism)
    
    Returns:
        List of product evaluation summaries
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ 目录不存在: {base_dir}")
        return []
    
    # Find all product directories
    product_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    product_dirs.sort()
    
    if not product_dirs:
        print(f"❌ 在 {base_dir} 中未找到产品目录")
        return []
    
    print(f"\n{'='*70}")
    print(f"🚀 开始批量评价")
    print(f"📁 基础目录: {base_dir}")
    print(f"📦 产品数量: {len(product_dirs)}")
    print(f"⚙️  每个产品最大并发: {max_workers_per_product}")
    print(f"⚙️  最大并发产品数: {max_concurrent_products}")
    print(f"{'='*70}\n")
    
    all_summaries = []
    
    # Process products with limited concurrency
    with ThreadPoolExecutor(max_workers=max_concurrent_products) as executor:
        # Submit all product tasks
        future_to_product = {
            executor.submit(
                evaluate_product_versions,
                product_dir,
                generate_screenshot=generate_screenshot,
                use_vision=use_vision,
                max_workers=max_workers_per_product,
                shuffle_versions=True,  # Shuffle to avoid order bias
                skip_if_summary_exists=skip_if_summary_exists  # Resume mechanism
            ): product_dir
            for product_dir in product_dirs
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_product):
            product_dir = future_to_product[future]
            completed += 1
            
            try:
                summary = future.result()
                all_summaries.append(summary)
                status = "✅" if summary.get('successful', 0) > 0 else "⚠️"
                print(f"\n{status} [{completed}/{len(product_dirs)}] {product_dir.name}: "
                      f"{summary.get('successful', 0)}/{summary.get('total_versions', 0)} 成功")
            except Exception as e:
                print(f"\n❌ [{completed}/{len(product_dirs)}] {product_dir.name}: 错误 - {e}")
                all_summaries.append({
                    'product_dir': str(product_dir),
                    'product_id': product_dir.name,
                    'error': str(e)
                })
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"🎉 批量评价完成！")
    print(f"{'='*70}")
    print(f"📊 总计: {len(all_summaries)} 个产品")
    
    successful_products = [s for s in all_summaries if s.get('successful', 0) > 0]
    if successful_products:
        total_versions = sum(s.get('total_versions', 0) for s in successful_products)
        total_successful = sum(s.get('successful', 0) for s in successful_products)
        avg_scores = [s.get('average_score', 0) for s in successful_products if 'average_score' in s]
        
        print(f"✅ 成功评价: {len(successful_products)} 个产品")
        print(f"📄 总版本数: {total_versions}")
        print(f"✅ 成功版本: {total_successful}")
        if avg_scores:
            print(f"📊 平均得分: {sum(avg_scores) / len(avg_scores):.2f}/10.0")
    
    return all_summaries


def batch_evaluate_posts(html_dir: str, 
                        output_file: Optional[str] = None,
                        generate_screenshot: bool = True,
                        use_vision: bool = True,
                        save_to_product_dir: bool = False) -> List[Dict]:
    """
    Batch evaluate multiple HTML posts
    
    Args:
        html_dir: Directory containing HTML files
        output_file: Optional output JSON file path (if None, saves to each product dir)
        generate_screenshot: Whether to generate screenshots
        use_vision: Whether to use vision model
        save_to_product_dir: Whether to save results to each product directory
    
    Returns:
        List of evaluation results
    """
    html_dir = Path(html_dir)
    html_files = list(html_dir.glob("*.html"))
    
    if not html_files:
        print(f"❌ 在 {html_dir} 中未找到HTML文件")
        return []
    
    results = []
    for i, html_file in enumerate(html_files, 1):
        print(f"\n{'='*60}")
        print(f"处理 {i}/{len(html_files)}: {html_file.name}")
        print(f"{'='*60}")
        try:
            result = evaluate_single_post(
                str(html_file), 
                generate_screenshot=generate_screenshot,
                use_vision=use_vision,
                save_to_product_dir=save_to_product_dir
            )
            results.append(result)
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'html_path': str(html_file),
                'error': str(e)
            })
    
    # Save aggregated results if output_file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 批量评价结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate HTML post quality")
    parser.add_argument("path", nargs='?', help="Path to HTML file, product directory, base directory, or shortcut (redbook/hupu)")
    parser.add_argument("--output", "-o", help="Output JSON file (for single file/dir mode)", default="post_evaluation.json")
    parser.add_argument("--no-screenshot", action="store_true", help="Don't generate screenshots")
    parser.add_argument("--no-vision", action="store_true", help="Don't use vision model")
    parser.add_argument("--batch-all", action="store_true", help="Batch evaluate all products in base directory")
    parser.add_argument("--max-workers", type=int, default=5, help="Max concurrent workers per product")
    parser.add_argument("--max-products", type=int, default=8, help="Max concurrent products")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip products with existing summary files (re-evaluate all)")
    
    args = parser.parse_args()
    
    # Handle shortcuts: redbook, hupu
    if args.path:
        if args.path.lower() == "redbook":
            args.path = "generated_redbook_it"
            args.batch_all = True
            print("📌 使用快捷方式: redbook -> generated_redbook_it --batch-all")
        elif args.path.lower() == "hupu":
            args.path = "generated_hupu_posts"
            args.batch_all = True
            print("📌 使用快捷方式: hupu -> generated_hupu_posts --batch-all")
    
    if not args.path:
        parser.print_help()
        sys.exit(1)
    
    path = Path(args.path)
    
    if args.batch_all:
        # Batch evaluate all products in base directory
        batch_evaluate_all_products(
            str(path),
            generate_screenshot=not args.no_screenshot,
            use_vision=not args.no_vision,
            max_workers_per_product=args.max_workers,
            max_concurrent_products=args.max_products,
            skip_if_summary_exists=not args.no_resume
        )
    elif path.is_file():
        # Single file
        result = evaluate_single_post(
            str(path),
            generate_screenshot=not args.no_screenshot,
            use_vision=not args.no_vision,
            save_to_product_dir=True
        )
        
        print(f"\n✅ 评价完成！")
        print(f"总体得分: {result['overall_score']:.2f}/10.0")
        
    elif path.is_dir():
        # Check if it's a product directory (has HTML files) or base directory
        html_files = list(path.glob("*.html"))
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        
        if html_files and not subdirs:
            # Product directory - evaluate all versions
            evaluate_product_versions(
                path,
                generate_screenshot=not args.no_screenshot,
                use_vision=not args.no_vision,
                max_workers=args.max_workers,
                shuffle_versions=True,
                skip_if_summary_exists=not args.no_resume
            )
        else:
            # Base directory - batch evaluate all products
            batch_evaluate_all_products(
                str(path),
                generate_screenshot=not args.no_screenshot,
                use_vision=not args.no_vision,
                max_workers_per_product=args.max_workers,
                max_concurrent_products=args.max_products,
                skip_if_summary_exists=not args.no_resume
            )
    else:
        print(f"❌ 路径不存在: {path}")

