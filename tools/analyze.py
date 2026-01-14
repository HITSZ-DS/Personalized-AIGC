import os
import json
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.video_analyst import VideoAnalyst
from agents.image_analyst import ImageAnalyst
from agents.text_analyst import TextAnalyst
from agents.user_profile_generator import UserProfileGenerator

def _convert_user_id(user_id, dataset):
    """Convert internal user_id to real user_id using user_map.json"""
    if dataset == 'bilibili':
        user_map_path = "SSLRec/datasets/general_cf/bilibili/user_map.json"
    elif dataset == 'douban':
        user_map_path = "SSLRec/datasets/general_cf/douban/user_map.json"
    elif dataset == 'redbook':
        user_map_path = "SSLRec/datasets/general_cf/redbook/user_map.json"
    elif dataset == 'hupu':
        user_map_path = "SSLRec/datasets/general_cf/hupu/user_map.json"
    else:
        # For unknown datasets, assume user_id is already real
        return str(user_id)

    try:
        with open(user_map_path, 'r', encoding='utf-8') as f:
            user_map = json.load(f)

        # user_id could be string or int
        real_user_id = user_map.get(str(user_id))
        if real_user_id:
            print(f"🔄 Converted user_id {user_id} -> {real_user_id}")
            return real_user_id
        else:
            print(f"⚠️  User_id {user_id} not found in user_map")
            return None

    except FileNotFoundError:
        print(f"⚠️  User mapping file not found: {user_map_path}")
        # Fallback: assume user_id is already real
        return str(user_id)
    except json.JSONDecodeError:
        print(f"⚠️  Invalid JSON in user mapping file: {user_map_path}")
        return str(user_id)

def get_file_type(filename):
    """Determine file type based on file extension"""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    video_extensions = ['.mp4']
    image_extensions = ['.jpg', '.jpeg', '.png']
    text_extensions = ['.txt']

    if ext in video_extensions:
        return "video"
    elif ext in image_extensions:
        return "image"
    elif ext in text_extensions:
        return "text"
    else:
        return "unknown"


def get_chronological_order(user_id, convert_user_id=True):
    """Get chronological order of items based on all CSV files for the user

    Args:
        user_id: User ID (could be internal or real depending on convert_user_id)
        convert_user_id: Whether to convert user_id to real_user_id (default: True for backward compatibility)
    """
    dataset = os.getenv("DATASET")
    if not dataset:
        print("⚠️  DATASET environment variable not set")
        return []

    # Convert internal user_id to real user_id if needed
    if convert_user_id:
        real_user_id = _convert_user_id(user_id, dataset)
        if not real_user_id:
            print(f"⚠️  Could not convert user_id {user_id} to real user_id")
            return []
        user_dataset_path = f"dataset/{dataset}/{real_user_id}"
        print(f"🔍 Processing dataset: {dataset} for user: {user_id} (real_id: {real_user_id})")
    else:
        # user_id is already real_user_id, no conversion needed
        user_dataset_path = f"dataset/{dataset}/{user_id}"
        print(f"🔍 Processing dataset: {dataset} for user: {user_id}")

    if not os.path.exists(user_dataset_path):
        print(f"⚠️  User dataset directory not found: {user_dataset_path}")
        return []

    try:
        # Dataset-specific column mapping
        dataset_config = {
            'bilibili': {
                'item_id_col': 'bvid',
                'time_col': 'fav_time',
                'required_cols': ['bvid', 'fav_time']
            },
            'douban': {
                'item_id_col': 'doubanID',  # Douban uses doubanID as unique identifier
                'time_col': 'fav_time',
                'required_cols': ['doubanID', 'fav_time']  # fav_time is optional for douban
            },
            'redbook': {
                'item_id_col': 'redbookID',  # RedBook uses redbookID as unique identifier
                'time_col': 'fav_time',
                'required_cols': ['redbookID', 'fav_time']  # fav_time is optional for redbook
            },
            'hupu': {
                'item_id_col': 'hupuID',  # Hupu uses hupuID as unique identifier
                'time_col': 'fav_time',
                'required_cols': ['hupuID', 'fav_time']  # fav_time is optional for hupu
            }
        }

        # Get configuration for current dataset
        config = dataset_config.get(dataset)
        if not config:
            print(f"⚠️  Unsupported dataset: {dataset}")
            return []

        item_id_col = config['item_id_col']
        time_col = config['time_col']
        required_cols = config['required_cols']

        # Find all CSV files in the user directory
        csv_files = []
        for file in os.listdir(user_dataset_path):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(user_dataset_path, file))

        if not csv_files:
            print(f"⚠️  No CSV files found in: {user_dataset_path}")
            return []

        print(f"📁 Found {len(csv_files)} CSV file(s) for user {user_id} in {dataset} dataset")

        # Read and combine all CSV files
        all_dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Check if required columns exist
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"  ⚠️  Missing columns {missing_cols} in {os.path.basename(csv_file)}")
                    continue

                # Extract only the required columns
                if time_col in df.columns:
                    df_subset = df[[item_id_col, time_col]].copy()
                    # Rename columns to standard names for processing
                    df_subset = df_subset.rename(columns={
                        item_id_col: 'item_id',
                        time_col: 'timestamp'
                    })
                else:
                    # If time column doesn't exist, create dummy timestamps
                    df_subset = df[[item_id_col]].copy()
                    df_subset = df_subset.rename(columns={item_id_col: 'item_id'})
                    # Use row index as dummy timestamp (reverse order for newest first)
                    df_subset['timestamp'] = range(len(df_subset)-1, -1, -1)
                    print(f"  ℹ️  Using dummy timestamps for {os.path.basename(csv_file)}")

                all_dataframes.append(df_subset)
                print(f"  ✅ Loaded {len(df_subset)} items from {os.path.basename(csv_file)}")

            except Exception as e:
                print(f"  ⚠️  Error reading {os.path.basename(csv_file)}: {e}")

        if not all_dataframes:
            print(f"⚠️  No valid CSV data found for user {user_id}")
            return []

        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Remove duplicates (same item might appear in multiple folders)
        combined_df = combined_df.drop_duplicates(subset=['item_id'])

        # Sort by timestamp in descending order (newest first)
        # Handle both numeric and non-numeric timestamps
        try:
            # Try to convert to numeric timestamps
            combined_df['timestamp'] = pd.to_numeric(combined_df['timestamp'], errors='coerce')
            # Fill NaN values with 0 for sorting
            combined_df['timestamp'] = combined_df['timestamp'].fillna(0)
            df_sorted = combined_df.sort_values('timestamp', ascending=False)
        except Exception as e:
            print(f"  ⚠️  Error sorting by timestamp: {e}, using original order")
            df_sorted = combined_df

        print(f"📅 Total {len(df_sorted)} unique items in chronological order for {dataset} dataset")

        # Return list of item_id in chronological order (newest to oldest)
        return df_sorted['item_id'].tolist()

    except Exception as e:
        print(f"⚠️  Error processing CSV files for user {user_id}: {e}")
        return []


def _load_item_mapping():
    """Load item mapping from item ID to title based on dataset type"""
    dataset = os.getenv("DATASET", "bilibili")

    # Configuration for dataset mapping files
    dataset_mapping_config = {
        'douban': {'file': 'douban_mapping.json', 'name': 'Douban'},
        'bilibili': {'file': 'bilibili_mapping.json', 'name': 'Bilibili'},
        'redbook': {'file': 'redbook_mapping.json', 'name': 'Redbook'},
        'hupu': {'file': 'hupu_mapping.json', 'name': 'Hupu'}
    }

    config = dataset_mapping_config.get(dataset)
    if not config:
        print(f"⚠️  No mapping file configured for dataset: {dataset}")
        return {}

    mapping_file = config['file']
    mapping_name = config['name']

    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            print(f"✅ Loaded {len(mapping)} items from {mapping_name} mapping")
            return mapping
        except Exception as e:
            print(f"⚠️  Error loading {mapping_name} mapping: {e}")
            return {}
    else:
        print(f"⚠️  {mapping_name} mapping file not found: {mapping_file}")
        return {}

def _get_item_title(item_id, item_mapping=None):
    """Get item title from item ID using mapping file (supports both old and new enhanced formats)"""
    if item_mapping is None:
        item_mapping = _load_item_mapping()

    dataset = os.getenv("DATASET", "bilibili")

    # Helper function to extract title from mapping data
    def extract_title(mapping_data):
        if isinstance(mapping_data, dict):
            # New enhanced format: {"title": "...", "content": "...", ...}
            return mapping_data.get('title', item_id)
        elif isinstance(mapping_data, str):
            # Old format: just the title string
            return mapping_data
        else:
            return item_id
    mapping_data = item_mapping.get(item_id)
    if mapping_data:
        return extract_title(mapping_data)

    # Fallback to item_id if no title found
    return item_id

def format_analysis_to_prompt(analysis_data, item_name, item_mapping=None):
    """Format analysis results into natural language prompt"""
    # Get title from item ID
    item_title = _get_item_title(item_name, item_mapping)
    prompt_parts = [f"## Note: {item_title}\n"]

    # Process text content
    if "text" in analysis_data and analysis_data["text"]:
        prompt_parts.append("### Text Content:")
        for idx, text_analysis in analysis_data["text"].items():
            prompt_parts.append(f"- {text_analysis}")
        prompt_parts.append("")

    # Process image content
    if "image" in analysis_data and analysis_data["image"]:
        prompt_parts.append("### Image Content:")
        for idx, image_analysis in analysis_data["image"].items():
            prompt_parts.append(f"- Image {idx}: {image_analysis}")
        prompt_parts.append("")

    # Process video content
    if "video" in analysis_data and analysis_data["video"]:
        prompt_parts.append("### Video Content:")
        for idx, video_analysis in analysis_data["video"].items():
            prompt_parts.append(f"- Video {idx}: {video_analysis}")
        prompt_parts.append("")

    return "\n".join(prompt_parts)


def _collect_all_items(dataset_path: str, max_users: int = None, skip_users: int = 0) -> tuple[List[Dict], List[str]]:
    """
    收集所有用户的所有item信息，跳过已完成的用户

    Args:
        dataset_path: Path to dataset directory
        max_users: Maximum number of users to process (None for all users)
        skip_users: Number of users to skip from the beginning (default: 0)

    Returns:
        Tuple of:
        - List of item info dicts with user_id, item_type, item_name, item_path, etc.
        - List of skipped user IDs (users with existing user_profile.txt)
    """
    all_item_tasks = []
    skipped_users = []
    processed_users = 0

    # Get all user directories and sort them for consistent ordering
    all_users = [user for user in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, user))]
    all_users.sort()  # Sort for consistent ordering

    # Skip the first N users if specified
    if skip_users > 0:
        all_users = all_users[skip_users:]
        print(f"⏭️  Skipping first {skip_users} users, starting from user {all_users[0] if all_users else 'N/A'}")

    for user in all_users:
        # Limit the number of users to process
        if max_users is not None and processed_users >= max_users:
            break

        user_dir = os.path.join(dataset_path, user)

        if not os.path.isdir(user_dir):
            continue

        # Check if user_profile.txt already exists - skip this user entirely
        user_profile_file = os.path.join(user_dir, "user_profile.txt")
        if os.path.exists(user_profile_file):
            print(f"⏭️  Skipping user {user} (user_profile.txt already exists)")
            skipped_users.append(user)
            continue

        # Increment processed users count (only count users that are actually processed)
        processed_users += 1

        # Process recommended items folder
        recommended_dir = os.path.join(user_dir, "recommended")
        if os.path.exists(recommended_dir):
            for item in os.listdir(recommended_dir):
                item_dir = os.path.join(recommended_dir, item)
                if os.path.isdir(item_dir):
                    all_item_tasks.append({
                        'user_id': user,
                        'item_type': 'recommended',
                        'item_name': item,
                        'item_path': item_dir
                    })

        # Process historical items folder
        historical_dir = os.path.join(user_dir, "historical")
        if os.path.exists(historical_dir):
            for item in os.listdir(historical_dir):
                item_dir = os.path.join(historical_dir, item)
                if os.path.isdir(item_dir):
                    all_item_tasks.append({
                        'user_id': user,
                        'item_type': 'historical',
                        'item_name': item,
                        'item_path': item_dir
                    })

        # Process validation items folder
        validation_dir = os.path.join(user_dir, "validation")
        if os.path.exists(validation_dir):
            for item in os.listdir(validation_dir):
                item_dir = os.path.join(validation_dir, item)
                if os.path.isdir(item_dir):
                    all_item_tasks.append({
                        'user_id': user,
                        'item_type': 'validation',
                        'item_name': item,
                        'item_path': item_dir
                    })

        # Process test items folder
        test_dir = os.path.join(user_dir, "test")
        if os.path.exists(test_dir):
            for item in os.listdir(test_dir):
                item_dir = os.path.join(test_dir, item)
                if os.path.isdir(item_dir):
                    all_item_tasks.append({
                        'user_id': user,
                        'item_type': 'test',
                        'item_name': item,
                        'item_path': item_dir
                    })

        # Handle legacy structure (items directly in user folder)
        for item in os.listdir(user_dir):
            item_dir = os.path.join(user_dir, item)

            # Skip if it's not a directory or if it's recommended/historical/validation/test folder
            if not os.path.isdir(item_dir) or item in ["recommended", "historical", "validation", "test"]:
                continue

            all_item_tasks.append({
                'user_id': user,
                'item_type': 'legacy',
                'item_name': item,
                'item_path': item_dir
            })

    return all_item_tasks, skipped_users

def _process_single_item_task(task: Dict, videoAnalyst, imageAnalyst, textAnalyst, item_mapping=None) -> Dict:
    """
    处理单个item任务

    Args:
        task: 包含用户ID、item类型、item名称和路径的字典
        item_mapping: Item ID到title的映射字典

    Returns:
        处理结果字典
    """
    user_id = task['user_id']
    item_type = task['item_type']
    item_name = task['item_name']
    item_path = task['item_path']

    try:
        print(f"  📁 Processing {user_id}/{item_type}/{item_name}")

        # 收集文件
        videos = []
        images = []
        text_list = []

        for file in os.listdir(item_path):
            file_type = get_file_type(file)
            if file_type == "video":
                videos.append(os.path.join(item_path, file))
            elif file_type == "image":
                images.append(os.path.join(item_path, file))
            elif file_type == "text":
                text_list.append(os.path.join(item_path, file))
            else:
                print(f"  ⚠️  Unknown file type: {file}")
                continue

        # 分析文件
        if videos:
            videoAnalyst(videos, item_path)
        if images:
            imageAnalyst(images, item_path)
        if text_list:
            textAnalyst(text_list, item_path)

        # 读取分析结果
        analysis_file = os.path.join(item_path, "analysis.json")
        analysis_data = None
        if os.path.exists(analysis_file):
            try:
                with open(analysis_file, "r", encoding="utf-8") as f:
                    analysis_data = json.loads(f.read())
            except Exception as e:
                print(f"  ⚠️  Error reading analysis file for {item_name}: {e}")

        # 格式化为prompt
        if analysis_data:
            formatted_prompt = format_analysis_to_prompt(analysis_data, item_name, item_mapping)
        elif videos or images or text_list:
            # 如果没有分析文件但有文件，创建基本信息
            item_title = _get_item_title(item_name, item_mapping)
            formatted_prompt = f"**Item: {item_title}**\n\nFiles found:\n- Videos: {len(videos)}\n- Images: {len(images)}\n- Texts: {len(text_list)}\n\nNote: Analysis in progress..."
        else:
            formatted_prompt = None

        result = {
            'user_id': user_id,
            'item_type': item_type,
            'item_name': item_name,
            'item_path': item_path,
            'formatted_prompt': formatted_prompt,
            'file_counts': {
                'videos': len(videos),
                'images': len(images),
                'texts': len(text_list)
            }
        }

        print(f"  ✅ Completed {user_id}/{item_type}/{item_name}: {len(videos)}v, {len(images)}i, {len(text_list)}t")
        return result

    except Exception as e:
        print(f"  ❌ Error processing {user_id}/{item_type}/{item_name}: {e}")
        return {
            'user_id': user_id,
            'item_type': item_type,
            'item_name': item_name,
            'error': str(e)
        }

def _process_single_user_profile(user_id: str, dataset_path: str, completed_tasks: List[Dict], userProfile) -> Dict:
    """
    Process a single user's profile generation

    Args:
        user_id: User ID (already real user ID, no conversion needed)
        dataset_path: Path to dataset directory
        completed_tasks: List of completed item analysis tasks
        userProfile: UserProfileGenerator instance

    Returns:
        Dict with processing results
    """
    try:
        print(f"\n🏠 Processing user profile: {user_id}")
        user_dir = os.path.join(dataset_path, user_id)

        # Check if user_profile.txt already exists (skip to avoid redundant generation)
        user_profile_file = os.path.join(user_dir, "user_profile.txt")
        if os.path.exists(user_profile_file):
            print(f"⏭️  Skipping user {user_id} (user_profile.txt already exists)")
            return {
                'user_id': user_id,
                'status': 'skipped',
                'reason': 'user_profile_exists'
            }

        # 获取时间顺序 (user_id is already real_user_id, no conversion needed)
        chronological_order = get_chronological_order(user_id, convert_user_id=False)
        print(f"📅 Found {len(chronological_order)} items in chronological order for {user_id}")

        # 收集这个用户的所有结果
        user_results = {
            'recommended': {},
            'historical': {},
            'validation': {},
            'test': {},
            'legacy': {}
        }

        for task_result in completed_tasks:
            if task_result['user_id'] == user_id and 'formatted_prompt' in task_result:
                item_type = task_result['item_type']
                item_name = task_result['item_name']
                formatted_prompt = task_result['formatted_prompt']

                if formatted_prompt:
                    user_results[item_type][item_name] = formatted_prompt

        # 按类型组织prompts
        recommended_prompts = list(user_results['recommended'].values())

        # Historical items需要按时间顺序排列
        historical_prompts = []
        if chronological_order:
            for item_id in chronological_order:
                if item_id in user_results['historical']:
                    historical_prompts.append(user_results['historical'][item_id])

            # 添加不在CSV中的历史items
            for item, prompt in user_results['historical'].items():
                if item not in chronological_order:
                    historical_prompts.append(prompt)
        else:
            historical_prompts = list(user_results['historical'].values())

        # Legacy items也需要按时间顺序排列
        legacy_prompts = []
        if chronological_order:
            for item_id in chronological_order:
                if item_id in user_results['legacy']:
                    legacy_prompts.append(user_results['legacy'][item_id])

            # 添加不在CSV中的legacy items
            for item, prompt in user_results['legacy'].items():
                if item not in chronological_order:
                    legacy_prompts.append(prompt)
        else:
            legacy_prompts = list(user_results['legacy'].values())

        # 组合所有prompts (不包含validation和test数据，只用于分析统计)
        dataset = os.getenv("DATASET", "unknown")
        sections = []

        if recommended_prompts:
            sections.append(f"## Recommended Items ({len(recommended_prompts)} items)\n\nThese are items recommended by the system for user {user_id} on the {dataset} platform.\n\n" + "\n---\n".join(recommended_prompts))

        if historical_prompts:
            sections.append(f"## Historical Interactions ({len(historical_prompts)} items)\n\nThese are items that user {user_id} has historically interacted with on the {dataset} platform, ordered chronologically from newest to oldest.\n\n" + "\n---\n".join(historical_prompts))

        if legacy_prompts:
            sections.append(f"## Legacy Items ({len(legacy_prompts)} items)\n\nThese items were processed using the legacy structure for user {user_id} on the {dataset} platform.\n\n" + "\n---\n".join(legacy_prompts))

        # 注意：validation和test数据不包含在prompt中，只用于统计
        validation_count = len(user_results['validation'])
        test_count = len(user_results['test'])

        # 创建最终的item profiles内容
        if sections:
            item_profiles = f"# All Content Analysis for User {user_id}\n\n" + "\n\n".join(sections)
        else:
            item_profiles = f"# All Content Analysis for User {user_id}\n\nNo items found for analysis."

        # 保存item profiles
        item_file = os.path.join(user_dir, "item_profiles.txt")
        with open(item_file, "w", encoding="utf-8") as f:
            f.write(item_profiles)

        print(f"✅ Item profiles saved for {user_id}: {item_file}")
        print(f"   - Recommended items: {len(recommended_prompts)}")
        print(f"   - Historical items: {len(historical_prompts)}")
        print(f"   - Validation items: {validation_count} (analyzed but not included in profile)")
        print(f"   - Test items: {test_count} (analyzed but not included in profile)")
        print(f"   - Legacy items: {len(legacy_prompts)}")

        # 生成用户画像
        print(f"🧠 Generating user profile for {user_id}...")
        user_profile = userProfile(item_profiles)
        user_file = os.path.join(user_dir, "user_profile.txt")
        with open(user_file, "w", encoding="utf-8") as f:
            f.write(user_profile)
        print(f"✅ User profile saved for {user_id}: {user_file}")

        return {
            'user_id': user_id,
            'status': 'completed',
            'recommended_items': len(recommended_prompts),
            'historical_items': len(historical_prompts),
            'validation_items': validation_count,
            'test_items': test_count,
            'legacy_items': len(legacy_prompts),
            'total_items': len(recommended_prompts) + len(historical_prompts) + len(legacy_prompts)  # validation和test不计入total，因为不用于生成profile
        }

    except Exception as e:
        print(f"❌ Error processing user profile for {user_id}: {e}")
        return {
            'user_id': user_id,
            'status': 'error',
            'error': str(e)
        }

def analyze(max_workers=15, user_profile_max_workers=15, max_users=None, skip_users=0):
    """
    使用流水线式并发分析所有用户的数据

    Pipeline strategy: 当某个用户的所有items分析完成后，立即开始生成该用户的profile
    这样可以让item分析和user profile生成并行进行，提高整体效率

    Args:
        max_workers: 最大并发worker数（用于item分析）
        user_profile_max_workers: 最大并发用户画像生成worker数
        max_users: 最大处理的用户数量（None表示处理所有用户）
        skip_users: 跳过前N个用户（默认0，从第一个用户开始）
    """
    print("🚀 Starting pipelined analysis for all users...")
    if skip_users > 0:
        print(f"⏭️  Skipping first {skip_users} users")
    if max_users:
        print(f"📌 Processing next {max_users} users")
    print("=" * 60)

    dataset_path = os.path.join("download", os.getenv("DATASET"))

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return

    # 初始化分析器
    videoAnalyst = VideoAnalyst(max_workers=2, video_max_workers=3)
    imageAnalyst = ImageAnalyst(max_workers=3)
    textAnalyst = TextAnalyst(max_workers=3)
    userProfile = UserProfileGenerator()

    # 加载item映射文件
    print("📋 Loading item mapping...")
    item_mapping = _load_item_mapping()

    # 1. 收集所有item任务（跳过已完成的用户）
    print("📊 Collecting all item tasks...")
    all_item_tasks, skipped_users = _collect_all_items(dataset_path, max_users=max_users, skip_users=skip_users)

    if skipped_users:
        print(f"⏭️  Skipped {len(skipped_users)} users with existing user_profile.txt")

    if not all_item_tasks:
        print("✅ No item tasks found (all users already processed)")
        return

    users = list(set(task['user_id'] for task in all_item_tasks))
    print(f"📈 Found {len(all_item_tasks)} item tasks across {len(users)} users to process")

    # 统计每个用户的item数量
    user_item_counts = defaultdict(int)
    for task in all_item_tasks:
        user_item_counts[task['user_id']] += 1

    print(f"📋 User item counts: {dict(user_item_counts)}")

    # 2. 使用流水线式并发处理
    print(f"🔄 Starting pipelined processing with {max_workers} item workers and {user_profile_max_workers} profile workers...")

    # 跟踪每个用户的完成状态
    user_completed_items = defaultdict(int)  # 每个用户已完成的item数量
    user_tasks_dict = defaultdict(list)  # 存储每个用户的completed tasks
    lock = Lock()  # 用于线程安全的操作

    # 存储所有结果
    completed_tasks = []
    user_profile_results = []
    user_profile_futures = {}  # 存储已提交的user profile任务

    # 创建两个线程池：一个用于item分析，一个用于user profile生成
    item_executor = ThreadPoolExecutor(max_workers=max_workers)
    profile_executor = ThreadPoolExecutor(max_workers=user_profile_max_workers)

    try:
        # 提交所有item任务
        future_to_task = {
            item_executor.submit(_process_single_item_task, task, videoAnalyst, imageAnalyst, textAnalyst, item_mapping): task
            for task in all_item_tasks
        }

        # 收集item分析结果，并在用户的items全部完成时触发profile生成
        completed_item_count = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            user_id = task['user_id']
            completed_item_count += 1

            try:
                result = future.result()

                with lock:
                    completed_tasks.append(result)
                    user_tasks_dict[user_id].append(result)
                    user_completed_items[user_id] += 1

                    current_completed = user_completed_items[user_id]
                    total_for_user = user_item_counts[user_id]

                    print(f"  ✅ [{completed_item_count}/{len(all_item_tasks)}] Completed item for user {user_id} ({current_completed}/{total_for_user})")

                    # 检查该用户的所有items是否都完成了
                    if current_completed == total_for_user and user_id not in user_profile_futures:
                        # 立即提交该用户的profile生成任务
                        print(f"  🎯 All items completed for user {user_id}, starting profile generation...")
                        profile_future = profile_executor.submit(
                            _process_single_user_profile,
                            user_id,
                            dataset_path,
                            user_tasks_dict[user_id],  # 只传递该用户的tasks
                            userProfile
                        )
                        user_profile_futures[user_id] = profile_future

            except Exception as e:
                print(f"  ❌ [{completed_item_count}/{len(all_item_tasks)}] Error processing task: {e}")
                error_result = {
                    'user_id': task['user_id'],
                    'item_type': task['item_type'],
                    'item_name': task['item_name'],
                    'error': str(e)
                }

                with lock:
                    completed_tasks.append(error_result)
                    user_tasks_dict[user_id].append(error_result)
                    user_completed_items[user_id] += 1

                    current_completed = user_completed_items[user_id]
                    total_for_user = user_item_counts[user_id]

                    # 即使有错误，也检查是否该启动profile生成
                    if current_completed == total_for_user and user_id not in user_profile_futures:
                        print(f"  🎯 All items processed for user {user_id} (with some errors), starting profile generation...")
                        profile_future = profile_executor.submit(
                            _process_single_user_profile,
                            user_id,
                            dataset_path,
                            user_tasks_dict[user_id],
                            userProfile
                        )
                        user_profile_futures[user_id] = profile_future

        # 所有item任务已完成
        print(f"\n✅ All item analysis completed!")
        print(f"⏳ Waiting for {len(user_profile_futures)} user profile generation tasks to complete...\n")

        # 收集所有user profile结果
        completed_profile_count = 0
        for user_id, future in user_profile_futures.items():
            completed_profile_count += 1
            try:
                result = future.result()
                user_profile_results.append(result)

                status_emoji = '✅' if result['status'] == 'completed' else '❌'
                print(f"  {status_emoji} [{completed_profile_count}/{len(user_profile_futures)}] User profile for {result['user_id']}: {result['status']}")

                if result['status'] == 'completed':
                    print(f"    📊 Items - Recommended: {result['recommended_items']}, Historical: {result['historical_items']}, Validation: {result['validation_items']}, Test: {result['test_items']}, Legacy: {result['legacy_items']}")

            except Exception as e:
                print(f"  ❌ [{completed_profile_count}/{len(user_profile_futures)}] Error processing user profile for {user_id}: {e}")
                user_profile_results.append({
                    'user_id': user_id,
                    'status': 'error',
                    'error': str(e)
                })

    finally:
        # 确保关闭线程池
        item_executor.shutdown(wait=True)
        profile_executor.shutdown(wait=True)

    print("\n" + "=" * 60)
    print("🎉 Pipelined analysis completed for all users!")
    print(f"📊 Summary:")
    print(f"  - Total users processed: {len(users)}")
    print(f"  - Total item tasks processed: {len(all_item_tasks)}")
    print(f"  - Successful item analyses: {len([r for r in completed_tasks if 'error' not in r])}")
    print(f"  - Failed item analyses: {len([r for r in completed_tasks if 'error' in r])}")
    print(f"  - Successful user profiles: {len([r for r in user_profile_results if r['status'] == 'completed'])}")
    print(f"  - Failed user profiles: {len([r for r in user_profile_results if r['status'] == 'error'])}")

    # 计算总的item统计
    total_recommended = sum(r.get('recommended_items', 0) for r in user_profile_results if r['status'] == 'completed')
    total_historical = sum(r.get('historical_items', 0) for r in user_profile_results if r['status'] == 'completed')
    total_validation = sum(r.get('validation_items', 0) for r in user_profile_results if r['status'] == 'completed')
    total_test = sum(r.get('test_items', 0) for r in user_profile_results if r['status'] == 'completed')
    total_legacy = sum(r.get('legacy_items', 0) for r in user_profile_results if r['status'] == 'completed')

    print(f"📈 Item Statistics:")
    print(f"  - Total recommended items: {total_recommended}")
    print(f"  - Total historical items: {total_historical}")
    print(f"  - Total validation items: {total_validation} (analyzed but not used for profile generation)")
    print(f"  - Total test items: {total_test} (analyzed but not used for profile generation)")
    print(f"  - Total legacy items: {total_legacy}")
    print(f"  - Items used for profile generation: {total_recommended + total_historical + total_legacy}")
    print(f"  - Grand total items analyzed: {total_recommended + total_historical + total_validation + total_test + total_legacy}")
    print("=" * 60)


