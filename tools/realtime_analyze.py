import os
import json
import pandas as pd
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.video_analyst import VideoAnalyst
from agents.image_analyst import ImageAnalyst
from agents.text_analyst import TextAnalyst


class RealtimeAnalyzer:
    """
    Analyzer for realtime data
    Processes realtime CSV data and downloaded videos to generate user profiles
    """

    def __init__(self, folder_max_workers=4):
        """Initialize the realtime analyzer

        Args:
            folder_max_workers: Maximum concurrent video folders to process
        """
        self.dataset = os.getenv("DATASET")
        self.realtime_dataset_path = f"dataset/{self.dataset}_realtime"
        self.realtime_download_path = f"download/{self.dataset}"

        # Concurrency control
        self.folder_max_workers = folder_max_workers

        # Initialize analysis agents with moderate concurrency to avoid file conflicts
        self.video_analyst = VideoAnalyst(max_workers=2, video_max_workers=3)
        self.image_analyst = ImageAnalyst(max_workers=2)
        self.text_analyst = TextAnalyst(max_workers=2)

    def get_realtime_users(self) -> List[str]:
        """
        Get all users that have realtime data

        Returns:
            List of user IDs with realtime data
        """
        users = []

        # Check realtime dataset directory
        if os.path.exists(self.realtime_dataset_path):
            for item in os.listdir(self.realtime_dataset_path):
                item_path = os.path.join(self.realtime_dataset_path, item)
                if os.path.isdir(item_path):
                    users.append(item)

        print(f"🔍 Found {len(users)} users with realtime data: {users}")
        return users

    def _collect_all_items(self, users: List[str]) -> List[Dict]:
        """
        收集所有用户的所有视频文件信息

        Returns:
            List of item info dicts with user_id, folder_name, item_files, etc.
        """
        all_item_tasks = []

        for user_id in users:
            user_download_path = os.path.join(self.realtime_download_path, user_id, "realtime")

            if not os.path.exists(user_download_path):
                continue

            item_folders = [d for d in os.listdir(user_download_path)
                           if os.path.isdir(os.path.join(user_download_path, d))]

            for item_folder in item_folders:
                item_folder_path = os.path.join(user_download_path, item_folder)

                # 收集这个文件夹中的所有文件
                video_files = []
                image_files = []
                text_files = []

                try:
                    for file in os.listdir(item_folder_path):
                        file_path = os.path.join(item_folder_path, file)
                        if os.path.isfile(file_path):
                            ext = os.path.splitext(file)[1].lower()
                            if ext in ['.mp4', '.avi', '.mkv', '.mov']:
                                video_files.append(file_path)
                            elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
                                image_files.append(file_path)
                            elif ext in ['.txt', '.json']:
                                text_files.append(file_path)

                    # 只有当有文件需要分析时才添加任务
                    if video_files or image_files or text_files:
                        all_item_tasks.append({
                            'user_id': user_id,
                            'folder_name': item_folder,
                            'folder_path': item_folder_path,
                            'video_files': video_files,
                            'image_files': image_files,
                            'text_files': text_files
                        })

                except Exception as e:
                    print(f"⚠️  Error scanning folder {item_folder_path}: {e}")
                    continue

        return all_item_tasks

    def _process_single_item_task(self, task: Dict) -> Dict:
        """
        处理单个item任务

        Args:
            task: 包含用户ID、文件夹信息和文件列表的字典

        Returns:
            处理结果字典
        """
        user_id = task['user_id']
        folder_name = task['folder_name']
        folder_path = task['folder_path']
        video_files = task['video_files']
        image_files = task['image_files']
        print("image_files", image_files)
        text_files = task['text_files']

        try:
            print(f"  🎬 Processing {user_id}/{folder_name}: {len(video_files)} videos, {len(image_files)} images, {len(text_files)} texts")

            # 分析文件
            if video_files:
                self.video_analyst(video_files, folder_path)

            if image_files:
                self.image_analyst(image_files, folder_path)

            if text_files:
                self.text_analyst(text_files, folder_path)

            # 读取分析结果
            analysis_data = {}
            analysis_file = os.path.join(folder_path, "analysis.json")
            if os.path.exists(analysis_file):
                try:
                    with open(analysis_file, "r", encoding="utf-8") as f:
                        analysis_data = json.load(f)
                except Exception as e:
                    print(f"  ⚠️  Error reading analysis file for {folder_name}: {e}")

            result = {
                'user_id': user_id,
                'folder_name': folder_name,
                'item_id': folder_name,
                'video_files': len(video_files),
                'image_files': len(image_files),
                'text_files': len(text_files),
                'analysis': analysis_data,
                'folder_path': folder_path
            }

            print(f"  ✅ Completed {user_id}/{folder_name}")
            return result

        except Exception as e:
            print(f"  ❌ Error processing {user_id}/{folder_name}: {e}")
            return {
                'user_id': user_id,
                'folder_name': folder_name,
                'error': str(e)
            }

    def __call__(self) -> Dict[str, Dict]:
        """
        使用全局视频池分析所有用户的realtime数据

        Returns:
            Dictionary mapping user_id to analysis results
        """
        print("🚀 Starting realtime analysis for all users with global item pool...")
        print("=" * 60)

        users = self.get_realtime_users()

        if not users:
            print("❌ No users with realtime data found")
            return {}

        # 1. 收集所有视频任务
        print("📊 Collecting all item tasks...")
        all_item_tasks = self._collect_all_items(users)

        if not all_item_tasks:
            print("❌ No item tasks found")
            return {}

        print(f"📈 Found {len(all_item_tasks)} item tasks across {len(users)} users")

        # 2. 使用全局进程池处理所有item
        print(f"🔄 Processing all items with {self.folder_max_workers} concurrent workers...")

        completed_tasks = []
        with ThreadPoolExecutor(max_workers=self.folder_max_workers) as executor:
            # 提交所有视频任务
            future_to_task = {
                executor.submit(self._process_single_item_task, task): task
                for task in all_item_tasks
            }

            # 收集结果
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1

                try:
                    result = future.result()
                    completed_tasks.append(result)
                    print(f"  ✅ [{completed}/{len(all_item_tasks)}] Completed task")

                except Exception as e:
                    print(f"  ❌ [{completed}/{len(all_item_tasks)}] Error processing task: {e}")
                    completed_tasks.append({
                        'user_id': task['user_id'],
                        'folder_name': task['folder_name'],
                        'error': str(e)
                    })

        # 3. 按用户整理结果
        print("📋 Organizing results by user...")
        results = {}

        for user_id in users:
            # 收集这个用户的item分析结果
            user_results = {}
            for task_result in completed_tasks:
                if task_result['user_id'] == user_id:
                    folder_name = task_result['folder_name']
                    user_results[folder_name] = {
                        'item_id': task_result.get('item_id', folder_name),
                        'video_files': task_result.get('video_files', 0),
                        'image_files': task_result.get('image_files', 0),
                        'text_files': task_result.get('text_files', 0),
                        'analysis': task_result.get('analysis', {}),
                        'folder_path': task_result.get('folder_path', ''),
                        'error': task_result.get('error')
                    }

            item_analysis = {
                "total_item_folders": len(user_results),
                "analyzed_items": user_results,
                "analysis_completed": len([r for r in user_results.values() if 'error' not in r])
            }

            results[user_id] = {
                "user_id": user_id,
                "item_analysis": item_analysis,
            }

            print(f"✅ Organized results for user {user_id}: {len(user_results)} folders")

        print("\n" + "=" * 60)
        print("🎉 Realtime analysis completed for all users!")
        print(f"📊 Summary:")
        print(f"  - Total users processed: {len(users)}")
        print(f"  - Total item tasks processed: {len(all_item_tasks)}")
        print(f"  - Successful analyses: {len([r for r in results.values() if 'error' not in r])}")
        print(f"  - Failed analyses: {len([r for r in results.values() if 'error' in r])}")
        print("=" * 60)

        return results

