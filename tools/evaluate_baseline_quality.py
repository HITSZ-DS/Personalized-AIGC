"""
Baseline Post Quality Evaluator

评估生成的 baseline 图文帖子（JPG 格式）的质量
评分维度：逻辑性、视觉呈现、拟人程度
"""

import os
import json
import sys
import base64
import time
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


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension"""
    dimension: str
    score: float  # 0-10


@dataclass
class BaselineEvaluationResult:
    """Complete evaluation result for a baseline post"""
    image_path: str
    overall_score: float  # Weighted average
    dimension_scores: List[DimensionScore]


class BaselineQualityEvaluator:
    """Evaluates baseline post quality using AI"""
    
    # Dimension weights for Redbook (sum to 1.0) - 3 dimensions for baseline
    DIMENSION_WEIGHTS_REDBOOK = {
        'logic': 0.30,              # 逻辑性
        'visual_presentation': 0.35, # 视觉呈现
        'human_likeness': 0.35,      # 拟人程度
    }
    
    # Dimension weights for Hupu (sum to 1.0) - 3 dimensions for baseline
    DIMENSION_WEIGHTS_HUPU = {
        'logic': 0.35,              # 逻辑性
        'visual_presentation': 0.35, # 视觉呈现
        'human_likeness': 0.30,      # 拟人程度
    }
    
    # Evaluation dimensions - 3 dimensions for baseline
    DIMENSIONS = {
        'logic': {
            'name': '逻辑性',
            'description': '内容结构是否清晰，逻辑是否连贯，信息是否有价值，文字与图片是否匹配（Hupu模式：文字与布局是否匹配）'
        },
        'visual_presentation': {
            'name': '视觉呈现',
            'description': '图片质量是否高，排版是否合理美观，视觉元素是否恰当，整体视觉效果是否吸引人（Hupu模式：排版是否合理美观，文字布局是否清晰）'
        },
        'human_likeness': {
            'name': '拟人程度',
            'description': '语言是否自然真实，是否有个人化表达，情感表达是否恰当，是否像真人写的'
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
        
        # Select dimension weights based on post type
        if self.post_type == 'hupu':
            self.DIMENSION_WEIGHTS = self.DIMENSION_WEIGHTS_HUPU
        else:
            self.DIMENSION_WEIGHTS = self.DIMENSION_WEIGHTS_REDBOOK
        
        # Initialize OpenAI client with .env configuration
        self.client = OpenAI(
            api_key=os.getenv("CHAT_API_KEY"),
            base_url=os.getenv("CHAT_BASE_URL"),
        )
        self.model = os.getenv("CHAT_MODEL", "claude-sonnet-4-5-20250929")
    
    def image_to_base64(self, image_path: str) -> Optional[str]:
        """Convert image to base64 string for API"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_str = base64.b64encode(image_data).decode('utf-8')
                
                # Determine MIME type from extension
                ext = Path(image_path).suffix.lower()
                mime_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.webp': 'image/webp'
                }.get(ext, 'image/jpeg')
                
                return f"data:{mime_type};base64,{base64_str}"
        except Exception as e:
            print(f"⚠️ Failed to convert image to base64: {e}")
            return None
    
    def create_evaluation_prompt(self) -> str:
        """Create prompt for AI evaluation"""
        prompt_parts = []
        
        # Detect platform name
        platform_name = "小红书" if self.post_type == 'redbook' else "虎扑论坛"
        content_type = "图文帖子" if self.post_type == 'redbook' else "讨论帖子"
        
        prompt_parts.append(f"你是一个专业的社交媒体内容评价专家。请对以下AI生成的{platform_name}风格{content_type}进行全面评价。")
        prompt_parts.append("\n" + "=" * 60)
        prompt_parts.append("评价维度（每个维度0-10分）：")
        prompt_parts.append("=" * 60)
        
        for dim_id, dim_info in self.DIMENSIONS.items():
            prompt_parts.append(f"\n{dim_info['name']} ({dim_id}):")
            prompt_parts.append(f"  {dim_info['description']}")
        
        prompt_parts.append("\n" + "=" * 60)
        prompt_parts.append("帖子内容：")
        prompt_parts.append("=" * 60)
        
        if self.post_type == 'redbook':
            prompt_parts.append("\n这是一张完整的小红书帖子截图，包含文字内容和图片。请仔细查看图片中的所有内容（包括文字、图片、布局等）进行评价。")
        else:
            prompt_parts.append("\n这是一张完整的虎扑论坛讨论帖子截图，主要包含文字内容（无图片）。请仔细查看图片中的所有内容（包括文字、布局、排版等）进行评价。")
        
        # Add evaluation instructions
        prompt_parts.append("\n" + "=" * 60)
        prompt_parts.append("评价要求：")
        prompt_parts.append("=" * 60)
        

# 评分要求——严（请严格执行）：
# 1. **评分标准（0-10分，保留1位小数）**：
#    - 9.0-10.0分：接近完美，几乎无可挑剔（极少数作品）
#    - 8.0-8.9分：优秀水平，有明显亮点但仍有小瑕疵
#    - 7.0-7.9分：良好水平，基本合格但有改进空间
#    - 6.0-6.9分：中等偏下，存在明显问题
#    - 5.0分以下：较差，有严重问题

# 2. **扣分参考标准（可弹性执行）**：
#    - 逻辑性：内容结构混乱-0.8分，逻辑不连贯-0.8分，信息价值低-0.5分，图文不匹配-0.8分
#    - 视觉呈现：图片质量差-0.5分，排版不美观-0.5分，元素不协调-0.5分，视觉吸引力弱-0.5分
#    - 拟人程度：语言生硬-0.5分，缺少个性化-0.5分，情感表达不自然-0.5分，AI痕迹明显-0.5分

# 3. **评分原则**：
#    - 从严评分，不要过于宽容
#    - 关注细节和质量，不要只看大体完整性
#    - 只有真正出色的作品才应给到8分以上

#  ---------------------------------------------------------

# 评分要求——中：
# 1. 对每个维度进行评分（0-10分，保留1位小数），请以高标准评价内容的完整性和质量
# 2. 特别关注内容是否存在需要改进的地方，评分时请严格把关
#    - 对于逻辑性问题、视觉缺陷、语言不自然等问题，应相应扣分
#    - 不要因为内容基本完整就给予高分，要关注细节和质量
#    - 评分应反映内容的真实质量水平，避免过于宽松
# 3. 计算加权平均分作为总体得分

#  ---------------------------------------------------------

# 评分要求（请保持合理宽容）——宽：

# 1. **评分标准（0-10分，保留1位小数）**：
#    - 8.0-10.0分：优秀到完美
#    - 7.0-7.9分：良好，大多数内容应在此区间
#    - 6.0-6.9分：基本合格
#    - 5.0分以下：存在明显严重问题

# 2. **各维度评分指引**：
#    - **逻辑性**（6-8分为主）：
#      * 内容基本连贯、有基本信息 → 6.5-7.0分
#      * 内容结构清晰、信息完整 → 7.5-8.0分
#      * 只有内容严重混乱、无法理解才低于6分
   
#    - **拟人程度**（6-8分为主）：
#      * 语言较为自然、表达流畅 → 6.5-7.0分
#      * 有个性化表达、情感自然 → 7.5-8.0分
#      * 只有语言极度生硬、明显机器痕迹才低于6分
   
#    - **视觉呈现**（7-8分为主）：
#      * 图片清晰、排版合理 → 7.0-7.5分
#      * 视觉吸引、设计精美 → 7.5-8.5分

# 3. **评分原则**：
#    - 保持合理宽容，不要过分苛刻
#    - 关注整体完成度，轻微瑕疵可以接受
#    - 大多数内容应在6.5-8.0分区间

        evaluation_instructions = """
评分要求（请保持合理宽容）：

1. **评分标准（0-10分，保留1位小数）**：
   - 8.0-10.0分：优秀到完美
   - 7.0-7.9分：良好，大多数内容应在此区间
   - 6.0-6.9分：基本合格
   - 5.0分以下：存在明显严重问题

2. **各维度评分指引**：
   - **逻辑性**（6-8分为主）：
     * 内容基本连贯、有基本信息 → 6.5-7.0分
     * 内容结构清晰、信息完整 → 7.5-8.0分
     * 只有内容严重混乱、无法理解才低于6分
   
   - **拟人程度**（6-8分为主）：
     * 语言较为自然、表达流畅 → 6.5-7.0分
     * 有个性化表达、情感自然 → 7.5-8.0分
     * 只有语言极度生硬、明显机器痕迹才低于6分
   
   - **视觉呈现**（7-8分为主）：
     * 图片清晰、排版合理 → 7.0-7.5分
     * 视觉吸引、设计精美 → 7.5-8.5分

3. **评分原则**：
   - 保持合理宽容，不要过分苛刻
   - 关注整体完成度，轻微瑕疵可以接受
   - 大多数内容应在6.5-8.0分区间
    
请以JSON格式输出结果，格式如下：
{
    "dimension_scores": [
        {
            "dimension": "logic",
            "score": 7.5
        },
        {
            "dimension": "visual_presentation",
            "score": 7.8
        },
        {
            "dimension": "human_likeness",
            "score": 7.2
        }
    ],
    "overall_score": 7.5
}

注意：只需要输出评分，不需要理由、建议或其他文字说明。
"""
        
        prompt_parts.append(evaluation_instructions)
        
        return "\n".join(prompt_parts)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def evaluate_with_ai(self, image_path: str) -> BaselineEvaluationResult:
        """
        Evaluate baseline post using AI
        
        Args:
            image_path: Path to JPG image file
        
        Returns:
            BaselineEvaluationResult object
        """
        # Create prompt
        prompt = self.create_evaluation_prompt()
        
        # Prepare content for API
        messages = [{"role": "user", "content": prompt}]
        
        # Add image if using vision model
        if self.use_vision_model:
            image_base64 = self.image_to_base64(image_path)
            if image_base64:
                messages[0]["content"] = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64}
                    }
                ]
        
        # Call AI API
        print("=" * 60)
        print("Calling AI API for evaluation...")
        print(f"Model: {self.model}")
        print(f"Using vision: {self.use_vision_model}")
        if isinstance(messages[0]["content"], list):
            has_image = any(c.get('type') == 'image_url' for c in messages[0]['content'])
            if has_image:
                print(f"Content: Text + Image")
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
            
            # Convert to BaselineEvaluationResult
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
            
            result = BaselineEvaluationResult(
                image_path=image_path,
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


def detect_post_type_from_path(path: str) -> str:
    """
    Detect post type from file path or directory name
    
    Args:
        path: File or directory path
    
    Returns:
        'redbook' or 'hupu'
    """
    path_str = str(path).lower()
    if 'hupu' in path_str or 'discussion_post' in path_str:
        return 'hupu'
    else:
        return 'redbook'


def evaluate_single_baseline(image_path: str, 
                             use_vision: bool = True,
                             post_type: Optional[str] = None) -> Dict:
    """
    Evaluate a single baseline JPG image
    
    Args:
        image_path: Path to JPG image file
        use_vision: Whether to use vision model
        post_type: 'redbook' or 'hupu'. If None, auto-detect from path
    
    Returns:
        Evaluation result dictionary
    """
    image_path_obj = Path(image_path)
    
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Auto-detect post type if not provided
    if post_type is None:
        post_type = detect_post_type_from_path(str(image_path_obj))
    
    # Create evaluator
    evaluator = BaselineQualityEvaluator(use_vision_model=use_vision, post_type=post_type)
    
    # Evaluate with AI
    print(f"\n{'='*60}")
    print(f"Evaluating: {image_path_obj.name}")
    print(f"{'='*60}\n")
    
    result = evaluator.evaluate_with_ai(str(image_path_obj))
    
    # Convert to dict for saving
    result_dict = {
        'image_path': str(image_path_obj),
        'image_filename': image_path_obj.name,
        'post_type': post_type,
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
    
    print(f"总体得分: {result.overall_score:.2f}/10.0")
    
    return result_dict


def find_baseline_images(product_dir: Path, post_type: str = 'redbook') -> List[Path]:
    """
    Find baseline JPG images in a product directory
    
    Args:
        product_dir: Product directory path
        post_type: 'redbook' or 'hupu' (determines file pattern)
    
    Returns:
        List of JPG image file paths
    """
    if post_type == 'hupu':
        # Look for discussion_post_*.jpg files (Hupu format)
        jpg_files = list(product_dir.glob("discussion_post_*.jpg"))
    else:
        # Look for it_product_*.jpg files (Redbook format)
        jpg_files = list(product_dir.glob("it_product_*.jpg"))
    
    # Also check for any .jpg files if no pattern-matched files found
    if not jpg_files:
        jpg_files = list(product_dir.glob("*.jpg"))
    
    return sorted(jpg_files, key=lambda x: x.name)


def evaluate_baseline_product(product_dir: Path,
                              use_vision: bool = True,
                              max_workers: int = 5,
                              skip_if_summary_exists: bool = True,
                              post_type: Optional[str] = None) -> Dict:
    """
    Evaluate baseline images in a single product directory
    
    Args:
        product_dir: Product directory path
        use_vision: Whether to use vision model
        max_workers: Maximum concurrent workers
        skip_if_summary_exists: Whether to skip if summary file already exists
        post_type: 'redbook' or 'hupu'. If None, auto-detect from directory path
    
    Returns:
        Dictionary with evaluation results
    """
    # Auto-detect post type if not provided
    if post_type is None:
        post_type = detect_post_type_from_path(str(product_dir))
    
    # Check if summary already exists (resume mechanism)
    summary_file = product_dir / "baseline_evaluation_summary.json"
    if skip_if_summary_exists and summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                existing_summary = json.load(f)
            print(f"\n{'='*60}")
            print(f"产品: {product_dir.name}")
            print(f"⏭️  已存在评价汇总，跳过评价")
            print(f"   平均得分: {existing_summary.get('average_score', 0):.2f}/10.0")
            print(f"   已评价图片: {existing_summary.get('successful', 0)}/{existing_summary.get('total_images', 0)}")
            print(f"{'='*60}")
            return existing_summary
        except Exception as e:
            print(f"⚠️  读取已有汇总文件失败: {e}，将重新评价")
    
    jpg_files = find_baseline_images(product_dir, post_type=post_type)
    
    if not jpg_files:
        return {
            'product_dir': str(product_dir),
            'product_id': product_dir.name,
            'images': [],
            'error': 'No JPG files found'
        }
    
    print(f"\n{'='*60}")
    print(f"产品: {product_dir.name}")
    print(f"类型: {post_type.upper()}")
    print(f"找到 {len(jpg_files)} 个图片文件")
    print(f"{'='*60}")
    
    results = {}
    errors = {}
    
    # Process files with limited concurrency to avoid API rate limits
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                evaluate_single_baseline,
                str(jpg_file),
                use_vision=use_vision,
                post_type=post_type
            ): jpg_file
            for jpg_file in jpg_files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            jpg_file = future_to_file[future]
            image_name = jpg_file.stem  # e.g., "it_product_0"
            
            try:
                result = future.result()
                results[image_name] = result
                print(f"  ✅ {jpg_file.name}: {result.get('overall_score', 0):.2f}/10.0")
            except Exception as e:
                error_msg = str(e)
                errors[image_name] = error_msg
                print(f"  ❌ {jpg_file.name}: {error_msg}")
    
    # Compile summary
    summary = {
        'product_dir': str(product_dir),
        'product_id': product_dir.name,
        'post_type': post_type,
        'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': len(jpg_files),
        'successful': len(results),
        'failed': len(errors),
        'images': {}
    }
    
    # Add image results
    for image_name, result in results.items():
        summary['images'][image_name] = {
            'image_file': result.get('image_filename', ''),
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
    
    # Calculate statistics
    if results:
        scores = [r.get('overall_score', 0) for r in results.values()]
        summary['average_score'] = sum(scores) / len(scores)
        summary['max_score'] = max(scores)
        summary['min_score'] = min(scores)
        
        # Find best image
        best_image = max(results.items(), key=lambda x: x[1].get('overall_score', 0))
        summary['best_image'] = {
            'name': best_image[0],
            'score': best_image[1].get('overall_score', 0)
        }
    
    # Save summary to product directory
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n  💾 评价汇总已保存到: {summary_file}")
    if results:
        print(f"  📊 平均得分: {summary['average_score']:.2f}/10.0")
        best_name = summary['best_image']['name']
        best_score = summary['best_image']['score']
        print(f"  🏆 最佳图片: {best_name} ({best_score:.2f}/10.0)")
    
    return summary


def batch_evaluate_all_baselines(base_dir: str,
                                 use_vision: bool = True,
                                 max_workers_per_product: int = 5,
                                 max_concurrent_products: int = 8,
                                 skip_if_summary_exists: bool = True,
                                 post_type: Optional[str] = None) -> List[Dict]:
    """
    Batch evaluate all baseline products in a directory
    
    Args:
        base_dir: Base directory containing product directories (e.g., generated_redbook_baseline or generated_hupu_baseline)
        use_vision: Whether to use vision model
        max_workers_per_product: Max concurrent images per product
        max_concurrent_products: Max concurrent products (to avoid overwhelming API)
        skip_if_summary_exists: Whether to skip products that already have summary files
        post_type: 'redbook' or 'hupu'. If None, auto-detect from base_dir name
    
    Returns:
        List of product evaluation summaries
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ 目录不存在: {base_dir}")
        return []
    
    # Auto-detect post type if not provided
    if post_type is None:
        post_type = detect_post_type_from_path(base_dir)
    
    # Find all product directories (format: {index}_{user_id})
    product_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    product_dirs.sort()
    
    if not product_dirs:
        print(f"❌ 在 {base_dir} 中未找到产品目录")
        return []
    
    # Check how many products already have evaluation summaries
    existing_summaries = []
    for product_dir in product_dirs:
        summary_file = product_dir / "baseline_evaluation_summary.json"
        if summary_file.exists():
            existing_summaries.append(product_dir)
    
    # Ask user if they want to re-evaluate all (if summaries exist and not forced to skip)
    should_re_evaluate = False
    if existing_summaries and skip_if_summary_exists:
        print(f"\n{'='*70}")
        print(f"📋 检测到已有评价汇总")
        print(f"{'='*70}")
        print(f"总产品数: {len(product_dirs)}")
        print(f"已有汇总: {len(existing_summaries)} 个产品")
        print(f"需要评价: {len(product_dirs) - len(existing_summaries)} 个产品")
        print(f"{'='*70}")
        
        # Ask user for confirmation (non-interactive mode: default to skip)
        try:
            user_input = input("\n❓ 是否重新评估全部产品？(y/n，默认n): ").strip().lower()
            if user_input in ['y', 'yes', '是', 'Y']:
                should_re_evaluate = True
                print("✅ 将重新评估全部产品（覆盖已有汇总）")
            else:
                print("⏭️  将跳过已有汇总的产品，只评估新产品")
        except (EOFError, KeyboardInterrupt):
            # Non-interactive mode (e.g., script running in background)
            print("⏭️  非交互模式：将跳过已有汇总的产品")
            should_re_evaluate = False
    
    # Update skip_if_summary_exists based on user choice
    if should_re_evaluate:
        skip_if_summary_exists = False
    
    print(f"\n{'='*70}")
    print(f"🚀 开始批量评价 Baseline {post_type.upper()} 帖子")
    print(f"📁 基础目录: {base_dir}")
    print(f"📦 产品数量: {len(product_dirs)}")
    print(f"⚙️  每个产品最大并发: {max_workers_per_product}")
    print(f"⚙️  最大并发产品数: {max_concurrent_products}")
    print(f"🔄 覆盖模式: {'是（重新评估全部）' if not skip_if_summary_exists else '否（跳过已有汇总）'}")
    print(f"{'='*70}\n")
    
    all_summaries = []
    
    # Process products with limited concurrency
    with ThreadPoolExecutor(max_workers=max_concurrent_products) as executor:
        # Submit all product tasks
        future_to_product = {
            executor.submit(
                evaluate_baseline_product,
                product_dir,
                use_vision=use_vision,
                max_workers=max_workers_per_product,
                skip_if_summary_exists=skip_if_summary_exists,
                post_type=post_type
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
                      f"{summary.get('successful', 0)}/{summary.get('total_images', 0)} 成功")
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
        total_images = sum(s.get('total_images', 0) for s in successful_products)
        total_successful = sum(s.get('successful', 0) for s in successful_products)
        avg_scores = [s.get('average_score', 0) for s in successful_products if 'average_score' in s]
        
        print(f"✅ 成功评价: {len(successful_products)} 个产品")
        print(f"📄 总图片数: {total_images}")
        print(f"✅ 成功图片: {total_successful}")
        if avg_scores:
            print(f"📊 平均得分: {sum(avg_scores) / len(avg_scores):.2f}/10.0")
    
    # Generate global statistics
    global_stats = generate_global_statistics(all_summaries, base_path)
    if global_stats:
        print(f"\n{'='*70}")
        print(f"📈 全局统计已保存")
        print(f"{'='*70}")
        print(f"📊 总分均值: {global_stats.get('overall_score_mean', 0):.2f}/10.0")
        print(f"📊 各维度均值:")
        for dim_id, dim_stat in global_stats.get('dimension_means', {}).items():
            dim_name = dim_stat.get('name', dim_id)
            dim_mean = dim_stat.get('mean', 0)
            print(f"   - {dim_name}: {dim_mean:.2f}/10.0")
        print(f"{'='*70}\n")
    
    return all_summaries


def generate_global_statistics(all_summaries: List[Dict], base_path: Path) -> Optional[Dict]:
    """
    Generate global statistics from all evaluation summaries
    
    Args:
        all_summaries: List of product evaluation summaries
        base_path: Base directory path for saving statistics
    
    Returns:
        Dictionary with global statistics, or None if no valid data
    """
    # Collect all dimension scores and overall scores
    dimension_scores_collection = {
        'logic': [],
        'visual_presentation': [],
        'human_likeness': []
    }
    overall_scores = []
    
    # Iterate through all products
    for summary in all_summaries:
        if 'images' not in summary:
            continue
        
        # Iterate through all images in this product
        for image_name, image_data in summary.get('images', {}).items():
            overall_score = image_data.get('overall_score', 0)
            if overall_score > 0:  # Only count valid scores
                overall_scores.append(overall_score)
            
            # Collect dimension scores
            dim_scores = image_data.get('dimension_scores', {})
            for dim_id in dimension_scores_collection.keys():
                if dim_id in dim_scores:
                    dim_score = dim_scores[dim_id].get('score', 0)
                    if dim_score > 0:  # Only count valid scores
                        dimension_scores_collection[dim_id].append(dim_score)
    
    # Calculate means
    if not overall_scores:
        print("⚠️  没有有效的评价数据，无法生成全局统计")
        return None
    
    # Overall score mean
    overall_mean = sum(overall_scores) / len(overall_scores)
    
    # Dimension means
    dimension_means = {}
    # Use redbook evaluator for dimension names (they're the same)
    evaluator = BaselineQualityEvaluator(post_type='redbook')
    for dim_id, scores in dimension_scores_collection.items():
        if scores:
            dim_mean = sum(scores) / len(scores)
            dim_name = evaluator.DIMENSIONS.get(dim_id, {}).get('name', dim_id)
            dimension_means[dim_id] = {
                'name': dim_name,
                'mean': dim_mean,
                'count': len(scores)
            }
    
    # Build statistics dictionary
    global_stats = {
        'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_products': len(all_summaries),
        'successful_products': len([s for s in all_summaries if s.get('successful', 0) > 0]),
        'total_images_evaluated': len(overall_scores),
        'overall_score_mean': overall_mean,
        'overall_score_max': max(overall_scores),
        'overall_score_min': min(overall_scores),
        'dimension_means': dimension_means,
        'dimension_statistics': {}
    }
    
    # Add detailed dimension statistics
    for dim_id, scores in dimension_scores_collection.items():
        if scores:
            dim_name = evaluator.DIMENSIONS.get(dim_id, {}).get('name', dim_id)
            global_stats['dimension_statistics'][dim_id] = {
                'name': dim_name,
                'mean': sum(scores) / len(scores),
                'max': max(scores),
                'min': min(scores),
                'count': len(scores)
            }
    
    # Save to JSON file
    stats_file = base_path / "baseline_global_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)
    
    print(f"💾 全局统计已保存到: {stats_file}")
    
    return global_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate baseline post quality (JPG images)")
    parser.add_argument("path", help="Path to JPG file, product directory, or base directory (generated_redbook_baseline or generated_hupu_baseline)")
    parser.add_argument("--no-vision", action="store_true", help="Don't use vision model")
    parser.add_argument("--batch-all", action="store_true", help="Batch evaluate all products in base directory")
    parser.add_argument("--max-workers", type=int, default=5, help="Max concurrent workers per product")
    parser.add_argument("--max-products", type=int, default=8, help="Max concurrent products")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip products with existing summary files (re-evaluate all)")
    parser.add_argument("--post-type", choices=['redbook', 'hupu'], help="Post type (auto-detected from path if not specified)")
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    # Auto-detect post type if not specified
    post_type = args.post_type
    if post_type is None:
        post_type = detect_post_type_from_path(str(path))
    
    if args.batch_all:
        # Batch evaluate all products in base directory
        batch_evaluate_all_baselines(
            str(path),
            use_vision=not args.no_vision,
            max_workers_per_product=args.max_workers,
            max_concurrent_products=args.max_products,
            skip_if_summary_exists=not args.no_resume,
            post_type=post_type
        )
    elif path.is_file():
        # Single file
        result = evaluate_single_baseline(
            str(path),
            use_vision=not args.no_vision,
            post_type=post_type
        )
        
        print(f"\n✅ 评价完成！")
        print(f"总体得分: {result['overall_score']:.2f}/10.0")
        
    elif path.is_dir():
        # Check if it's a product directory (has JPG files) or base directory
        jpg_files = list(path.glob("*.jpg"))
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        
        if jpg_files and not subdirs:
            # Product directory - evaluate all images
            evaluate_baseline_product(
                path,
                use_vision=not args.no_vision,
                max_workers=args.max_workers,
                skip_if_summary_exists=not args.no_resume,
                post_type=post_type
            )
        else:
            # Base directory - batch evaluate all products
            batch_evaluate_all_baselines(
                str(path),
                use_vision=not args.no_vision,
                max_workers_per_product=args.max_workers,
                max_concurrent_products=args.max_products,
                skip_if_summary_exists=not args.no_resume,
                post_type=post_type
            )
    else:
        print(f"❌ 路径不存在: {path}")

