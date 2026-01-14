"""
Post Evaluation Statistics Analyzer

Analyzes evaluation results across all products and versions:
- Overall score statistics by version
- Score improvement between versions
- Best/worst versions
- Platform comparison (redbook vs hupu)
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class VersionStats:
    """Statistics for a single version"""
    version_name: str
    count: int
    avg_score: float
    max_score: float
    min_score: float
    scores: List[float]


@dataclass
class VersionComparison:
    """Comparison between two versions"""
    from_version: str
    to_version: str
    improvement_count: int
    decline_count: int
    no_change_count: int
    avg_improvement: float
    max_improvement: float
    min_improvement: float


def load_evaluation_summaries(base_dir: str, is_baseline: bool = False) -> List[Dict]:
    """
    Load all evaluation summaries from product directories
    
    Args:
        base_dir: Base directory (generated_redbook_it, generated_hupu_posts, or generated_redbook_baseline)
        is_baseline: Whether this is baseline evaluation (uses baseline_evaluation_summary.json)
    
    Returns:
        List of evaluation summary dictionaries
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ 目录不存在: {base_dir}")
        return []
    
    summaries = []
    product_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # Determine summary file name
    if is_baseline:
        summary_filename = "baseline_evaluation_summary.json"
    else:
        summary_filename = "post_evaluations_summary.json"
    
    for product_dir in product_dirs:
        summary_file = product_dir / summary_filename
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    # Mark as baseline for later processing
                    if is_baseline:
                        summary['_is_baseline'] = True
                    summaries.append(summary)
            except Exception as e:
                print(f"⚠️  读取失败 {summary_file}: {e}")
    
    # Auto-detect baseline format if no summaries found with standard format
    if not summaries and not is_baseline:
        baseline_summaries = []
        for product_dir in product_dirs:
            baseline_file = product_dir / "baseline_evaluation_summary.json"
            if baseline_file.exists():
                try:
                    with open(baseline_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                        summary['_is_baseline'] = True
                        baseline_summaries.append(summary)
                except Exception as e:
                    print(f"⚠️  读取失败 {baseline_file}: {e}")
        
        if baseline_summaries:
            print(f"ℹ️  检测到 baseline 格式，使用 baseline_evaluation_summary.json")
            return baseline_summaries
    
    return summaries


def extract_version_scores(summaries: List[Dict]) -> Dict[str, List[float]]:
    """
    Extract scores by version name (for regular evaluations) or image name (for baseline)
    
    Args:
        summaries: List of evaluation summaries
    
    Returns:
        Dictionary mapping version/image name to list of scores
    """
    version_scores = defaultdict(list)
    
    for summary in summaries:
        if summary.get('_is_baseline', False):
            # Baseline format: extract from images
            images = summary.get('images', {})
            for image_name, image_data in images.items():
                score = image_data.get('overall_score', 0)
                if score > 0:  # Only include valid scores
                    version_scores[image_name].append(score)
        else:
            # Regular format: extract from versions
            versions = summary.get('versions', {})
            for version_name, version_data in versions.items():
                score = version_data.get('overall_score', 0)
                if score > 0:  # Only include valid scores
                    version_scores[version_name].append(score)
    
    return dict(version_scores)


def calculate_version_stats(version_scores: Dict[str, List[float]]) -> Dict[str, VersionStats]:
    """
    Calculate statistics for each version
    
    Args:
        version_scores: Dictionary mapping version name to list of scores
    
    Returns:
        Dictionary mapping version name to VersionStats
    """
    stats = {}
    
    for version_name, scores in version_scores.items():
        if scores:
            stats[version_name] = VersionStats(
                version_name=version_name,
                count=len(scores),
                avg_score=sum(scores) / len(scores),
                max_score=max(scores),
                min_score=min(scores),
                scores=scores
            )
    
    return stats


def extract_version_sequence(version_name: str) -> Tuple[int, bool]:
    """
    Extract version number from version name
    
    Args:
        version_name: e.g., "image_text_v0", "discussion_post_v5", "image_text_english"
    
    Returns:
        Tuple of (version_number, is_english)
        - version_number: -1 for english, 0+ for v0, v1, etc.
        - is_english: True if english version
    """
    if 'english' in version_name.lower():
        return -1, True
    
    # Extract version number
    import re
    match = re.search(r'v(\d+)', version_name)
    if match:
        return int(match.group(1)), False
    
    return -2, False  # Unknown version


def calculate_version_improvements(summaries: List[Dict]) -> List[VersionComparison]:
    """
    Calculate score improvements between consecutive numeric versions only
    (e.g., v0->v1, v1->v2), excluding english versions
    
    Args:
        summaries: List of evaluation summaries
    
    Returns:
        List of VersionComparison objects
    """
    comparisons = []
    
    for summary in summaries:
        versions = summary.get('versions', {})
        if len(versions) < 2:
            continue
        
        # Filter and sort only numeric versions (exclude english)
        version_items = []
        for version_name, version_data in versions.items():
            version_num, is_english = extract_version_sequence(version_name)
            score = version_data.get('overall_score', 0)
            # Only include numeric versions (not english)
            if score > 0 and not is_english and version_num >= 0:
                version_items.append((version_num, version_name, score))
        
        # Sort by version number: v0, v1, v2, ...
        version_items.sort(key=lambda x: x[0])
        
        # Compare only consecutive numeric versions
        for i in range(len(version_items) - 1):
            from_num, from_name, from_score = version_items[i]
            to_num, to_name, to_score = version_items[i + 1]
            
            # Only compare if versions are consecutive (v0->v1, v1->v2, etc.)
            if to_num != from_num + 1:
                continue  # Skip non-consecutive versions
            
            improvement = to_score - from_score
            
            # Find or create comparison
            comp = next((c for c in comparisons if c.from_version == from_name and c.to_version == to_name), None)
            
            if comp is None:
                comp = VersionComparison(
                    from_version=from_name,
                    to_version=to_name,
                    improvement_count=0,
                    decline_count=0,
                    no_change_count=0,
                    avg_improvement=0.0,
                    max_improvement=float('-inf'),
                    min_improvement=float('inf'),
                )
                comparisons.append(comp)
            
            # Update statistics
            if improvement > 0.01:  # Significant improvement
                comp.improvement_count += 1
            elif improvement < -0.01:  # Significant decline
                comp.decline_count += 1
            else:
                comp.no_change_count += 1
            
            comp.max_improvement = max(comp.max_improvement, improvement)
            comp.min_improvement = min(comp.min_improvement, improvement)
    
    # Calculate average improvements
    for comp in comparisons:
        total = comp.improvement_count + comp.decline_count + comp.no_change_count
        if total > 0:
            # Recalculate from all improvements
            improvements = []
            for summary in summaries:
                versions = summary.get('versions', {})
                if comp.from_version in versions and comp.to_version in versions:
                    from_score = versions[comp.from_version].get('overall_score', 0)
                    to_score = versions[comp.to_version].get('overall_score', 0)
                    if from_score > 0 and to_score > 0:
                        improvements.append(to_score - from_score)
            
            if improvements:
                comp.avg_improvement = sum(improvements) / len(improvements)
    
    return comparisons


def recalculate_with_baseline_weights(version_data: Dict, platform_type: str) -> Optional[float]:
    """
    Recalculate product score using baseline dimension weights
    
    For Redbook: exclude platform_fit, use weights (logic 0.30, visual_presentation 0.35, human_likeness 0.35)
    For Hupu: exclude link_relevance and platform_fit, use weights (logic 0.35, visual_presentation 0.35, human_likeness 0.30)
    
    Args:
        version_data: Version data dictionary with dimension_scores
        platform_type: 'redbook' or 'hupu'
    
    Returns:
        Recalculated overall score, or None if required dimensions are missing
    """
    if platform_type == 'hupu':
        # Baseline dimension weights for hupu
        BASELINE_WEIGHTS = {
            'logic': 0.35,
            'visual_presentation': 0.35,
            'human_likeness': 0.30
        }
        
        # Dimension mapping: hupu regular → baseline
        DIMENSION_MAPPING = {
            'content_quality': 'logic',  # content_quality maps to logic
            'link_relevance': 'visual_presentation',  # link_relevance maps to visual_presentation
            'human_likeness': 'human_likeness',  # same
            # platform_fit is excluded
        }
    else:  # redbook
        # Baseline dimension weights for redbook
        BASELINE_WEIGHTS = {
            'logic': 0.30,
            'visual_presentation': 0.35,
            'human_likeness': 0.35
        }
        
        # Dimension mapping: redbook regular → baseline
        DIMENSION_MAPPING = {
            'content_quality': 'logic',  # content_quality maps to logic
            'visual_presentation': 'visual_presentation',  # same
            'human_likeness': 'human_likeness',  # same
            # platform_fit is excluded
        }
    
    dim_scores = version_data.get('dimension_scores', {})
    if not dim_scores:
        return None
    
    # Extract and map dimensions
    baseline_scores = {}
    for regular_dim, baseline_dim in DIMENSION_MAPPING.items():
        if regular_dim in dim_scores:
            dim_data = dim_scores[regular_dim]
            if isinstance(dim_data, dict):
                score = dim_data.get('score', 0)
            else:
                score = float(dim_data) if dim_data else 0
            
            if score > 0:
                baseline_scores[baseline_dim] = score
    
    # Check if we have all required dimensions (at least 2 out of 3)
    if len(baseline_scores) < 2:
        return None  # Missing too many required dimensions
    
    # Calculate weighted score (normalize by actual weights used)
    total_score = 0.0
    total_weight = 0.0
    
    for baseline_dim, score in baseline_scores.items():
        weight = BASELINE_WEIGHTS.get(baseline_dim, 0.0)
        total_score += score * weight
        total_weight += weight
    
    if total_weight == 0:
        return None
    
    # Normalize by total weight used (in case some dimensions are missing)
    return total_score / total_weight


def recalculate_summaries_with_baseline_weights(summaries: List[Dict], platform_type: str) -> List[Dict]:
    """
    Recalculate all product scores using baseline dimension weights
    
    Args:
        summaries: List of evaluation summaries
        platform_type: 'redbook' or 'hupu'
    
    Returns:
        List of summaries with recalculated scores (new field: baseline_recalculated_score)
    """
    recalculated_summaries = []
    
    for summary in summaries:
        # Only process regular evaluations (not baseline format)
        if summary.get('_is_baseline', False):
            recalculated_summaries.append(summary)
            continue
        
        # Check if this has versions
        versions = summary.get('versions', {})
        if not versions:
            recalculated_summaries.append(summary)
            continue
        
        # Check first version to see if it has content_quality dimension
        first_version = next(iter(versions.values()), {})
        dim_scores = first_version.get('dimension_scores', {})
        
        # If doesn't have content_quality, skip (not applicable)
        if 'content_quality' not in dim_scores:
            recalculated_summaries.append(summary)
            continue
        
        # Recalculate each version
        new_summary = summary.copy()
        new_versions = {}
        
        for version_name, version_data in versions.items():
            new_version_data = version_data.copy()
            baseline_score = recalculate_with_baseline_weights(version_data, platform_type)
            
            if baseline_score is not None:
                new_version_data['baseline_recalculated_score'] = round(baseline_score, 2)
            
            new_versions[version_name] = new_version_data
        
        new_summary['versions'] = new_versions
        
        # Recalculate product average (excluding english versions)
        baseline_scores = []
        for version_name, version_data in new_versions.items():
            if 'english' not in version_name.lower():
                baseline_score = version_data.get('baseline_recalculated_score')
                if baseline_score is not None:
                    baseline_scores.append(baseline_score)
        
        if baseline_scores:
            new_summary['baseline_recalculated_average'] = round(sum(baseline_scores) / len(baseline_scores), 2)
        
        recalculated_summaries.append(new_summary)
    
    return recalculated_summaries


def extract_dimension_scores(summaries: List[Dict]) -> Dict[str, List[float]]:
    """
    Extract dimension scores across all evaluations
    
    Args:
        summaries: List of evaluation summaries
    
    Returns:
        Dictionary mapping dimension name to list of scores
    """
    dimension_scores = defaultdict(list)
    
    for summary in summaries:
        if summary.get('_is_baseline', False):
            # Baseline format: extract from images
            images = summary.get('images', {})
            for image_data in images.values():
                dim_scores = image_data.get('dimension_scores', {})
                for dim_id, dim_data in dim_scores.items():
                    if isinstance(dim_data, dict):
                        score = dim_data.get('score', 0)
                    else:
                        score = float(dim_data) if dim_data else 0
                    if score > 0:
                        dimension_scores[dim_id].append(score)
        else:
            # Regular format: extract from versions
            versions = summary.get('versions', {})
            for version_data in versions.values():
                dim_scores = version_data.get('dimension_scores', {})
                for dim_id, dim_data in dim_scores.items():
                    if isinstance(dim_data, dict):
                        score = dim_data.get('score', 0)
                    else:
                        score = float(dim_data) if dim_data else 0
                    if score > 0:
                        dimension_scores[dim_id].append(score)
    
    return dict(dimension_scores)


def calculate_best_version_scores(summaries: List[Dict], is_baseline_format: bool) -> List[float]:
    """
    Calculate best version score for each product
    
    Args:
        summaries: List of evaluation summaries
        is_baseline_format: Whether this is baseline format
    
    Returns:
        List of best version scores (one per product)
    """
    best_scores = []
    
    for summary in summaries:
        if is_baseline_format:
            # Baseline format: get max score from images
            images = summary.get('images', {})
            if images:
                scores = [img_data.get('overall_score', 0) for img_data in images.values() if img_data.get('overall_score', 0) > 0]
                if scores:
                    best_scores.append(max(scores))
        else:
            # Regular format: get max score from versions (excluding english)
            versions = summary.get('versions', {})
            if versions:
                scores = []
                for version_name, version_data in versions.items():
                    # Exclude english versions from best version calculation
                    if 'english' not in version_name.lower():
                        score = version_data.get('overall_score', 0)
                        if score > 0:
                            scores.append(score)
                if scores:
                    best_scores.append(max(scores))
    
    return best_scores


def extract_v3_scores(summaries: List[Dict], platform_type: str = 'redbook') -> Tuple[List[float], Dict[str, List[float]], List[float], Dict[str, List[float]]]:
    """
    Extract v3 version scores (overall and dimension scores)
    
    Args:
        summaries: List of evaluation summaries
        platform_type: 'redbook' or 'hupu'
    
    Returns:
        Tuple of (overall_scores, dimension_scores_dict, baseline_recalculated_scores, baseline_dimension_scores)
        - overall_scores: List of overall scores for v3 versions
        - dimension_scores_dict: Dictionary mapping dimension ID to list of scores (original 4 dimensions)
        - baseline_recalculated_scores: List of baseline recalculated scores for v3 versions
        - baseline_dimension_scores: Dictionary mapping baseline dimension ID to list of scores (3 dimensions: logic, visual_presentation, human_likeness)
    """
    overall_scores = []
    dimension_scores = defaultdict(list)
    baseline_recalculated_scores = []
    baseline_dimension_scores = defaultdict(list)
    
    # Dimension mapping based on platform type
    if platform_type == 'hupu':
        # Hupu: content_quality → logic, link_relevance → visual_presentation
        DIMENSION_MAPPING = {
            'content_quality': 'logic',
            'link_relevance': 'visual_presentation',
            'human_likeness': 'human_likeness'
            # platform_fit is excluded
        }
    else:
        # Redbook: content_quality → logic, visual_presentation stays same
        DIMENSION_MAPPING = {
            'content_quality': 'logic',
            'visual_presentation': 'visual_presentation',
            'human_likeness': 'human_likeness'
            # platform_fit is excluded
        }
    
    for summary in summaries:
        versions = summary.get('versions', {})
        for version_name, version_data in versions.items():
            version_num, is_english = extract_version_sequence(version_name)
            
            # Only extract v3 (version number == 3)
            if version_num == 3 and not is_english:
                overall_score = version_data.get('overall_score', 0)
                if overall_score > 0:
                    overall_scores.append(overall_score)
                
                # Extract baseline recalculated score if available
                baseline_score = version_data.get('baseline_recalculated_score')
                if baseline_score is not None:
                    baseline_recalculated_scores.append(baseline_score)
                
                # Extract dimension scores (original)
                dim_scores = version_data.get('dimension_scores', {})
                for dim_id, dim_data in dim_scores.items():
                    if isinstance(dim_data, dict):
                        score = dim_data.get('score', 0)
                    else:
                        score = float(dim_data) if dim_data else 0
                    if score > 0:
                        dimension_scores[dim_id].append(score)
                        
                        # Map to baseline dimensions (exclude platform_fit)
                        if dim_id in DIMENSION_MAPPING:
                            baseline_dim = DIMENSION_MAPPING[dim_id]
                            baseline_dimension_scores[baseline_dim].append(score)
    
    return overall_scores, dict(dimension_scores), baseline_recalculated_scores, dict(baseline_dimension_scores)


def calculate_best_version_dimension_scores(summaries: List[Dict], is_baseline_format: bool) -> Dict[str, List[float]]:
    """
    Calculate dimension scores for best version of each product
    
    Args:
        summaries: List of evaluation summaries
        is_baseline_format: Whether this is baseline format
    
    Returns:
        Dictionary mapping dimension ID to list of scores (one per product's best version)
    """
    dimension_scores = defaultdict(list)
    
    for summary in summaries:
        if is_baseline_format:
            # Baseline format: get best image (max overall_score)
            images = summary.get('images', {})
            if images:
                best_image = None
                best_score = -1
                for image_name, image_data in images.items():
                    score = image_data.get('overall_score', 0)
                    if score > best_score:
                        best_score = score
                        best_image = image_data
                
                if best_image:
                    dim_scores = best_image.get('dimension_scores', {})
                    for dim_id, dim_data in dim_scores.items():
                        if isinstance(dim_data, dict):
                            score = dim_data.get('score', 0)
                        else:
                            score = float(dim_data) if dim_data else 0
                        if score > 0:
                            dimension_scores[dim_id].append(score)
        else:
            # Regular format: get best version (excluding english)
            versions = summary.get('versions', {})
            if versions:
                best_version = None
                best_score = -1
                for version_name, version_data in versions.items():
                    # Exclude english versions
                    if 'english' not in version_name.lower():
                        score = version_data.get('overall_score', 0)
                        if score > best_score:
                            best_score = score
                            best_version = version_data
                
                if best_version:
                    dim_scores = best_version.get('dimension_scores', {})
                    for dim_id, dim_data in dim_scores.items():
                        if isinstance(dim_data, dict):
                            score = dim_data.get('score', 0)
                        else:
                            score = float(dim_data) if dim_data else 0
                        if score > 0:
                            dimension_scores[dim_id].append(score)
    
    return dict(dimension_scores)


def generate_statistics_report(base_dir: str, output_file: Optional[str] = None, is_baseline: Optional[bool] = None) -> Dict:
    """
    Generate comprehensive statistics report in JSON format
    
    Args:
        base_dir: Base directory (generated_redbook_it, generated_hupu_posts, generated_redbook_baseline, or generated_hupu_baseline)
        output_file: Output JSON file path (if None, saves to base_dir)
        is_baseline: Whether this is baseline evaluation (None for auto-detect)
    
    Returns:
        Report dictionary (JSON-serializable)
    """
    # Auto-detect baseline format
    if is_baseline is None:
        # Check if any baseline files exist
        base_path = Path(base_dir)
        if base_path.exists():
            product_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            baseline_count = sum(1 for d in product_dirs if (d / "baseline_evaluation_summary.json").exists())
            standard_count = sum(1 for d in product_dirs if (d / "post_evaluations_summary.json").exists())
            is_baseline = baseline_count > standard_count
    
    # Load all summaries
    summaries = load_evaluation_summaries(base_dir, is_baseline=is_baseline if is_baseline is not None else False)
    
    if not summaries:
        error_msg = f"❌ 未找到任何评价汇总文件在 {base_dir}"
        print(error_msg)
        return {"error": error_msg}
    
    # Check if baseline format
    is_baseline_format = summaries[0].get('_is_baseline', False) if summaries else False
    
    # Detect platform type from directory name
    platform_type = "unknown"
    if "hupu" in base_dir.lower():
        platform_type = "hupu"
    elif "redbook" in base_dir.lower():
        platform_type = "redbook"
    
    # Recalculate products with baseline weights (if not baseline format)
    if not is_baseline_format and platform_type in ['redbook', 'hupu']:
        platform_name = "小红书" if platform_type == 'redbook' else "虎扑"
        print(f"🔄 重新计算 {platform_name} 产品得分（使用 Baseline 维度权重）...")
        summaries = recalculate_summaries_with_baseline_weights(summaries, platform_type)
    
    # Extract version/image scores
    version_scores = extract_version_scores(summaries)
    version_stats = calculate_version_stats(version_scores)
    
    # Extract baseline recalculated scores (for redbook regular evaluations)
    baseline_recalculated_scores = []
    baseline_recalculated_version_scores = defaultdict(list)
    if not is_baseline_format:
        for summary in summaries:
            versions = summary.get('versions', {})
            for version_name, version_data in versions.items():
                baseline_score = version_data.get('baseline_recalculated_score')
                if baseline_score is not None:
                    baseline_recalculated_scores.append(baseline_score)
                    # Exclude english versions from version-level stats
                    if 'english' not in version_name.lower():
                        baseline_recalculated_version_scores[version_name].append(baseline_score)
    
    # Extract dimension scores
    dimension_scores = extract_dimension_scores(summaries)
    
    # Calculate improvements (only for regular evaluations, not baseline)
    improvements = []
    if not is_baseline_format:
        improvements = calculate_version_improvements(summaries)
    
    # Collect all scores (all versions/images)
    all_scores = []
    for summary in summaries:
        if is_baseline_format:
            # Baseline format: extract from images
            images = summary.get('images', {})
            for image_data in images.values():
                score = image_data.get('overall_score', 0)
                if score > 0:
                    all_scores.append(score)
        else:
            # Regular format: extract from versions
            versions = summary.get('versions', {})
            for version_data in versions.values():
                score = version_data.get('overall_score', 0)
                if score > 0:
                    all_scores.append(score)
    
    # Calculate best version scores (one per product)
    best_version_scores = calculate_best_version_scores(summaries, is_baseline_format)
    
    # Calculate best version dimension scores (one per product)
    best_version_dimension_scores = calculate_best_version_dimension_scores(summaries, is_baseline_format)
    
    # Extract v3 scores (only for regular evaluations, not baseline)
    v3_overall_scores = []
    v3_dimension_scores = {}
    v3_baseline_recalculated_scores = []
    v3_baseline_dimension_scores = {}
    if not is_baseline_format:
        v3_overall_scores, v3_dimension_scores, v3_baseline_recalculated_scores, v3_baseline_dimension_scores = extract_v3_scores(summaries, platform_type)
    
    # Build JSON report structure
    if is_baseline_format:
        if platform_type == "hupu":
            report_title = "Baseline 虎扑讨论帖评价统计报告"
        else:
            report_title = "Baseline 小红书图文评价统计报告"
    else:
        if platform_type == "hupu":
            report_title = "虎扑帖子评价统计报告"
        else:
            report_title = "小红书帖子评价统计报告"
    
    # Build report structure based on format type
    if is_baseline_format:
        report = {
            "report_title": report_title,
            "generation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_directory": base_dir,
            "platform_type": platform_type,
            "total_products": len(summaries),
            "evaluation_type": "Baseline (图片)",
            "overall_statistics": {},
            "v3_statistics": {},
            "dimension_statistics": {},
            "best_and_worst_products": {}
        }
    else:
        report = {
            "report_title": report_title,
            "generation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_directory": base_dir,
            "platform_type": platform_type,
            "total_products": len(summaries),
            "evaluation_type": "常规 (版本迭代)",
            "overall_statistics": {},
            "v3_statistics": {},
            "dimension_statistics": {},
            "version_statistics": {},
            "best_and_worst_products": {}
        }
    
    # Overall statistics
    if all_scores:
        report["overall_statistics"] = {
            "total_evaluations": len(all_scores),
            "average_score_all_versions": round(sum(all_scores) / len(all_scores), 2),
            "max_score": round(max(all_scores), 2),
            "min_score": round(min(all_scores), 2)
        }
        
        # Add best version average (only for regular evaluations with multiple versions)
        if best_version_scores and not is_baseline_format:
            report["overall_statistics"]["average_score_best_versions"] = round(sum(best_version_scores) / len(best_version_scores), 2)
            report["overall_statistics"]["best_version_count"] = len(best_version_scores)
            report["overall_statistics"]["note"] = "average_score_all_versions: 所有产品的所有版本平均分; average_score_best_versions: 每个产品最高分版本的平均分"
        elif is_baseline_format:
            # For baseline, best version average equals all versions average (usually one image per product)
            report["overall_statistics"]["average_score_best_versions"] = round(sum(all_scores) / len(all_scores), 2)
            report["overall_statistics"]["note"] = "Baseline格式通常每个产品只有一个图片，所以两个平均值相同"
        
        # Add baseline recalculated statistics (for regular evaluations)
        if baseline_recalculated_scores and not is_baseline_format:
            if platform_type == 'hupu':
                note = "使用Baseline维度权重重新计算（content_quality→logic, link_relevance→visual_presentation, 舍弃platform_fit，权重：logic 0.35, visual_presentation 0.35, human_likeness 0.30）"
            else:  # redbook
                note = "使用Baseline维度权重重新计算（content_quality→logic, 舍弃platform_fit，权重：logic 0.30, visual_presentation 0.35, human_likeness 0.35）"
            
            report["baseline_recalculated_statistics"] = {
                "total_evaluations": len(baseline_recalculated_scores),
                "average_score_all_versions": round(sum(baseline_recalculated_scores) / len(baseline_recalculated_scores), 2),
                "max_score": round(max(baseline_recalculated_scores), 2),
                "min_score": round(min(baseline_recalculated_scores), 2),
                "note": note
            }
            
            # Calculate best version average with baseline weights
            baseline_best_scores = []
            for summary in summaries:
                baseline_avg = summary.get('baseline_recalculated_average')
                if baseline_avg is not None:
                    baseline_best_scores.append(baseline_avg)
            
            if baseline_best_scores:
                report["baseline_recalculated_statistics"]["average_score_best_versions"] = round(sum(baseline_best_scores) / len(baseline_best_scores), 2)
                report["baseline_recalculated_statistics"]["best_version_count"] = len(baseline_best_scores)
    
    # V3 version statistics (only for regular evaluations) - placed before dimension_statistics
    dim_name_map = {
        'logic': '逻辑性',
        'visual_presentation': '视觉呈现',
        'human_likeness': '拟人程度',
        'content_quality': '内容质量',
        'platform_fit': '平台适配性',
        'link_relevance': '链接相关性'
    }
    
    if not is_baseline_format and v3_overall_scores:
        report["v3_statistics"] = {
            "overall": {
                "evaluation_count": len(v3_overall_scores),
                "average_score": round(sum(v3_overall_scores) / len(v3_overall_scores), 2),
                "max_score": round(max(v3_overall_scores), 2),
                "min_score": round(min(v3_overall_scores), 2)
            },
            "dimensions": {}
        }
        
        # Add baseline recalculated statistics for v3
        if v3_baseline_recalculated_scores:
            if platform_type == 'hupu':
                v3_note = "使用Baseline维度权重重新计算（content_quality→logic, link_relevance→visual_presentation, 舍弃platform_fit，权重：logic 0.35, visual_presentation 0.35, human_likeness 0.30）"
            else:  # redbook
                v3_note = "使用Baseline维度权重重新计算（content_quality→logic, 舍弃platform_fit，权重：logic 0.30, visual_presentation 0.35, human_likeness 0.35）"
            
            report["v3_statistics"]["baseline_recalculated"] = {
                "evaluation_count": len(v3_baseline_recalculated_scores),
                "average_score": round(sum(v3_baseline_recalculated_scores) / len(v3_baseline_recalculated_scores), 2),
                "max_score": round(max(v3_baseline_recalculated_scores), 2),
                "min_score": round(min(v3_baseline_recalculated_scores), 2),
                "note": v3_note,
                "dimensions": {}
            }
            
            # Add baseline dimension statistics for v3 (3 dimensions only)
            if v3_baseline_dimension_scores:
                for dim_id, scores in sorted(v3_baseline_dimension_scores.items()):
                    dim_name = dim_name_map.get(dim_id, dim_id)
                    if scores:
                        report["v3_statistics"]["baseline_recalculated"]["dimensions"][dim_id] = {
                            "name": dim_name,
                            "evaluation_count": len(scores),
                            "average_score": round(sum(scores) / len(scores), 2),
                            "max_score": round(max(scores), 2),
                            "min_score": round(min(scores), 2)
                        }
        
        # Add dimension statistics for v3 (original 4 dimensions)
        if v3_dimension_scores:
            for dim_id, scores in sorted(v3_dimension_scores.items()):
                dim_name = dim_name_map.get(dim_id, dim_id)
                if scores:
                    report["v3_statistics"]["dimensions"][dim_id] = {
                        "name": dim_name,
                        "evaluation_count": len(scores),
                        "average_score": round(sum(scores) / len(scores), 2),
                        "max_score": round(max(scores), 2),
                        "min_score": round(min(scores), 2)
                    }
    
    # Dimension statistics
    if dimension_scores:
        report["dimension_statistics"] = {}
        for dim_id, scores in sorted(dimension_scores.items()):
            dim_name = dim_name_map.get(dim_id, dim_id)
            if scores:
                dim_stat = {
                    "name": dim_name,
                    "evaluation_count": len(scores),
                    "average_score": round(sum(scores) / len(scores), 2),
                    "max_score": round(max(scores), 2),
                    "min_score": round(min(scores), 2)
                }
                
                # Add best version dimension average if available
                if dim_id in best_version_dimension_scores and best_version_dimension_scores[dim_id]:
                    best_dim_scores = best_version_dimension_scores[dim_id]
                    dim_stat["average_score_best_versions"] = round(sum(best_dim_scores) / len(best_dim_scores), 2)
                    dim_stat["best_version_count"] = len(best_dim_scores)
                    # Different notes for baseline vs regular format
                    if is_baseline_format:
                        dim_stat["note"] = "Baseline格式每个产品通常只有一张图片"
                    else:
                        dim_stat["note"] = "average_score: 所有版本的该维度平均分; average_score_best_versions: 每个产品最佳版本的该维度平均分"
                
                report["dimension_statistics"][dim_id] = dim_stat
    
    # Version/Image statistics
    # Add baseline recalculated version statistics (for regular evaluations only)
    if baseline_recalculated_version_scores and not is_baseline_format:
        baseline_version_stats = calculate_version_stats(baseline_recalculated_version_scores)
        report["baseline_recalculated_version_statistics"] = {}
        
        # Sort versions: v0, v1, v2, ..., english
        sorted_versions = sorted(baseline_version_stats.items(), key=lambda x: (
            extract_version_sequence(x[0])[1],  # english first
            extract_version_sequence(x[0])[0]   # then by number
        ))
        
        for version_name, stats in sorted_versions:
            report["baseline_recalculated_version_statistics"][version_name] = {
                "type": "version",
                "evaluation_count": stats.count,
                "average_score": round(stats.avg_score, 2),
                "max_score": round(stats.max_score, 2),
                "min_score": round(stats.min_score, 2)
            }
    
    # Add version or image statistics
    if is_baseline_format:
        # For baseline, show all images in "image_statistics"
        report["image_statistics"] = {}
        if version_stats:
            for image_name, stats in sorted(version_stats.items()):
                report["image_statistics"][image_name] = {
                    "type": "image",
                    "evaluation_count": stats.count,
                    "average_score": round(stats.avg_score, 2),
                    "max_score": round(stats.max_score, 2),
                    "min_score": round(stats.min_score, 2)
                }
    else:
        # For regular evaluations, show versions in "version_statistics"
        report["version_statistics"] = {}
        # Sort versions: v0, v1, v2, ..., english
        sorted_versions = sorted(version_stats.items(), key=lambda x: (
            extract_version_sequence(x[0])[1],  # english first
            extract_version_sequence(x[0])[0]   # then by number
        ))
        
        for version_name, stats in sorted_versions:
            report["version_statistics"][version_name] = {
                "type": "version",
                "evaluation_count": stats.count,
                "average_score": round(stats.avg_score, 2),
                "max_score": round(stats.max_score, 2),
                "min_score": round(stats.min_score, 2)
            }
    
    # Version improvements (only for regular evaluations)
    if not is_baseline_format:
        report["version_improvements"] = {}
        if improvements:
            # Sort improvements by from_version (only numeric versions)
            sorted_improvements = sorted(improvements, key=lambda x: (
                extract_version_sequence(x.from_version)[1],  # is_english (should be False)
                extract_version_sequence(x.from_version)[0]    # version number
            ))
            
            for comp in sorted_improvements:
                total = comp.improvement_count + comp.decline_count + comp.no_change_count
                if total == 0:
                    continue
                comparison_key = f"{comp.from_version} → {comp.to_version}"
                report["version_improvements"][comparison_key] = {
                    "total_comparisons": total,
                    "average_change": round(comp.avg_improvement, 2),
                    "improvements": comp.improvement_count,
                    "improvements_percentage": round(comp.improvement_count / total * 100, 1),
                    "declines": comp.decline_count,
                    "declines_percentage": round(comp.decline_count / total * 100, 1),
                    "no_change": comp.no_change_count,
                    "no_change_percentage": round(comp.no_change_count / total * 100, 1),
                    "max_improvement": round(comp.max_improvement, 2),
                    "max_decline": round(comp.min_improvement, 2)
                }
    
    # Best and worst products
    product_scores = []
    for summary in summaries:
        avg_score = summary.get('average_score', 0)
        if avg_score > 0:
            product_scores.append({
                "product_id": summary.get('product_id', 'unknown'),
                "score": round(avg_score, 2)
            })
    
    if product_scores:
        product_scores.sort(key=lambda x: x['score'], reverse=True)
        report["best_and_worst_products"] = {
            "top_10": product_scores[:10],
            "bottom_10": product_scores[-10:]
        }
    
    # Version distribution (only for regular evaluations, not baseline)
    if not is_baseline_format:
        version_counts = defaultdict(int)
        for summary in summaries:
            versions = summary.get('versions', {})
            for version_name in versions.keys():
                version_counts[version_name] += 1
        
        report["version_distribution"] = {}
        for version_name, count in sorted(version_counts.items(), key=lambda x: (
            extract_version_sequence(x[0])[1],
            extract_version_sequence(x[0])[0]
        )):
            report["version_distribution"][version_name] = count
    else:
        # For baseline, add a note that there are no versions
        report["note_on_versions"] = "Baseline格式没有版本迭代概念，每个产品只有单张图片评估"
    
    # Save to JSON file
    if output_file is None:
        if is_baseline_format:
            output_file = Path(base_dir) / "baseline_evaluation_statistics.json"
        else:
            output_file = Path(base_dir) / "evaluation_statistics.json"
    else:
        output_file = Path(output_file)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 统计报告已保存到: {output_file}")
    
    return report


def compare_platforms(redbook_dir: str, hupu_dir: str, output_file: Optional[str] = None):
    """
    Compare statistics between Redbook and Hupu platforms
    
    Args:
        redbook_dir: Redbook base directory
        hupu_dir: Hupu base directory
        output_file: Output text file path
    """
    redbook_summaries = load_evaluation_summaries(redbook_dir)
    hupu_summaries = load_evaluation_summaries(hupu_dir)
    
    lines = []
    lines.append("=" * 80)
    lines.append("平台对比统计报告")
    lines.append("=" * 80)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Redbook statistics
    lines.append("=" * 80)
    lines.append("小红书 (Redbook)")
    lines.append("=" * 80)
    if redbook_summaries:
        redbook_scores = []
        for summary in redbook_summaries:
            avg = summary.get('average_score', 0)
            if avg > 0:
                redbook_scores.append(avg)
        
        if redbook_scores:
            lines.append(f"产品数量: {len(redbook_summaries)}")
            lines.append(f"平均得分: {sum(redbook_scores) / len(redbook_scores):.2f}/10.0")
            lines.append(f"最高得分: {max(redbook_scores):.2f}/10.0")
            lines.append(f"最低得分: {min(redbook_scores):.2f}/10.0")
    else:
        lines.append("无数据")
    lines.append("")
    
    # Hupu statistics
    lines.append("=" * 80)
    lines.append("虎扑 (Hupu)")
    lines.append("=" * 80)
    if hupu_summaries:
        hupu_scores = []
        for summary in hupu_summaries:
            avg = summary.get('average_score', 0)
            if avg > 0:
                hupu_scores.append(avg)
        
        if hupu_scores:
            lines.append(f"产品数量: {len(hupu_summaries)}")
            lines.append(f"平均得分: {sum(hupu_scores) / len(hupu_scores):.2f}/10.0")
            lines.append(f"最高得分: {max(hupu_scores):.2f}/10.0")
            lines.append(f"最低得分: {min(hupu_scores):.2f}/10.0")
    else:
        lines.append("无数据")
    lines.append("")
    
    # Comparison
    if redbook_summaries and hupu_summaries:
        lines.append("=" * 80)
        lines.append("对比分析")
        lines.append("=" * 80)
        redbook_avg = sum(redbook_scores) / len(redbook_scores) if redbook_scores else 0
        hupu_avg = sum(hupu_scores) / len(hupu_scores) if hupu_scores else 0
        
        if redbook_avg > hupu_avg:
            lines.append(f"小红书平均得分更高: {redbook_avg:.2f} vs {hupu_avg:.2f} (+{redbook_avg - hupu_avg:.2f})")
        elif hupu_avg > redbook_avg:
            lines.append(f"虎扑平均得分更高: {hupu_avg:.2f} vs {redbook_avg:.2f} (+{hupu_avg - redbook_avg:.2f})")
        else:
            lines.append(f"两个平台平均得分相同: {redbook_avg:.2f}")
    
    lines.append("")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    if output_file is None:
        output_file = Path("evaluation_platform_comparison.txt")
    else:
        output_file = Path(output_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ 平台对比报告已保存到: {output_file}")
    
    return report_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate post evaluation statistics")
    parser.add_argument("base_dir", nargs='?', help="Base directory or shortcut (redbook/hupu/redbook_baseline/hupu_baseline)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--baseline", action="store_true", help="Force baseline format (auto-detect if not specified)")
    parser.add_argument("--compare", action="store_true", help="Compare Redbook and Hupu (requires both directories)")
    parser.add_argument("--redbook-dir", help="Redbook directory for comparison")
    parser.add_argument("--hupu-dir", help="Hupu directory for comparison")
    
    args = parser.parse_args()
    
    # Handle shortcuts
    if args.base_dir:
        base_dir_lower = args.base_dir.lower()
        if base_dir_lower == "redbook":
            args.base_dir = "generated_redbook_it"
            print("📌 使用快捷方式: redbook -> generated_redbook_it")
        elif base_dir_lower == "hupu":
            args.base_dir = "generated_hupu_posts"
            print("📌 使用快捷方式: hupu -> generated_hupu_posts")
        elif base_dir_lower in ["redbook_baseline", "redbookbaseline", "baseline"]:
            args.base_dir = "generated_redbook_baseline"
            args.baseline = True  # Auto-set baseline flag
            print("📌 使用快捷方式: redbook_baseline -> generated_redbook_baseline (baseline格式)")
        elif base_dir_lower in ["hupu_baseline", "hupubaseline", "comment_baseline", "commentbaseline"]:
            args.base_dir = "generated_hupu_baseline"
            args.baseline = True  # Auto-set baseline flag
            print("📌 使用快捷方式: hupu_baseline -> generated_hupu_baseline (baseline格式)")
    
    if not args.base_dir:
        parser.print_help()
        print("\n快捷方式:")
        print("  python stat_post_evaluations.py redbook         # generated_redbook_it")
        print("  python stat_post_evaluations.py hupu            # generated_hupu_posts")
        print("  python stat_post_evaluations.py redbook_baseline  # generated_redbook_baseline")
        print("  python stat_post_evaluations.py hupu_baseline   # generated_hupu_baseline")
        sys.exit(1)
    
    if args.compare:
        redbook_dir = args.redbook_dir or "generated_redbook_it"
        hupu_dir = args.hupu_dir or "generated_hupu_posts"
        compare_platforms(redbook_dir, hupu_dir, args.output)
    else:
        is_baseline = args.baseline if args.baseline else None
        generate_statistics_report(args.base_dir, args.output, is_baseline=is_baseline)

