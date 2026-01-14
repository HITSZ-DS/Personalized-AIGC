
import asyncio
import json
import os
import torch
import subprocess
import sys
import random
import time
import pickle
import requests
import re
from typing import Optional, List, Dict
from config.configurator import configs

# 设置随机种子以确保结果可复现
torch.manual_seed(2025)


class BilibiliDownloader:
    """
    Bilibili video downloader with configurable limits

    Usage example:
        downloader = BilibiliDownloader(model)
        # Download with default limits (1 hour max duration, 3 videos max per playlist)
        await downloader.bilibili_test()

        # Download with custom limits (2 hours max duration, 5 videos max per playlist)
        await downloader.bilibili_test(max_duration_hours=2, max_videos_per_playlist=5)
    """
    def __init__(self, model):
        self.model = model

    async def download_bilibili_video(self, bvid, save_path, max_retries=3, max_duration_hours=0.1, max_videos_per_playlist=1):
        """Download bilibili video with retry mechanism and anti-bot measures

        Args:
            bvid: Bilibili video ID
            save_path: Path to save the video
            max_retries: Maximum retry attempts
            max_duration_hours: Maximum video duration in hours (videos longer than this will be truncated)
            max_videos_per_playlist: Maximum number of videos to download from a playlist/collection
        """
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid being detected as bot
                if attempt > 0:
                    delay = random.uniform(2, 3)  # 5-10 seconds delay
                    print(f"[INFO] Retry {attempt} for {bvid}, waiting {delay:.1f} seconds...")
                    await asyncio.sleep(delay)

                url = f"https://www.bilibili.com/video/{bvid}"
                print(f"[INFO] Downloading Bilibili video: {bvid} to {save_path} (Attempt {attempt + 1}/{max_retries})", flush=True)
                print(f"[INFO] 限制设置: 最大时长 {max_duration_hours}h, 播放列表最多 {max_videos_per_playlist} 个视频", flush=True)

                # Enhanced yt-dlp command with better options and duration/count limits
                cmd = [
                    "yt-dlp",
                    "--output", f"{save_path}/%(title)s.%(ext)s",
                    "--no-warnings",
                    "--retries", "3",
                    "--fragment-retries", "3",
                    "--skip-unavailable-fragments",
                    "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "--referer", "https://www.bilibili.com/",
                ]

                # Add duration limit (convert hours to seconds for yt-dlp)
                max_duration_seconds = max_duration_hours * 3600
                cmd.extend(["--download-sections", f"*0-{max_duration_seconds}"])

                # Add playlist item limit
                cmd.extend(["--playlist-end", str(max_videos_per_playlist)])

                # Add the URL at the end
                cmd.append(url)

                # Run yt-dlp command asynchronously
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    print(f"[SUCCESS] Download completed: {bvid}", flush=True)
                    return True
                else:
                    try:
                        error_msg = stderr.decode('utf-8')
                    except UnicodeDecodeError:
                        error_msg = stderr.decode('gbk', errors='ignore')

                    # Check if it's a permanent error (no need to retry)
                    if any(keyword in error_msg.lower() for keyword in ['private', 'deleted', 'not available', 'geo-blocked']):
                        print(f"[ERROR] Permanent failure for {bvid}: {error_msg}")
                        return False

                    print(f"[WARNING] Attempt {attempt + 1} failed for {bvid}: {error_msg}")

                    if attempt == max_retries - 1:
                        print(f"[ERROR] All attempts failed for {bvid}")
                        return False

            except Exception as e:
                print(f"[ERROR] Exception during download of {bvid}: {e}")
                if attempt == max_retries - 1:
                    return False

        return False

    def get_user_validation_items(self, user_internal_id: int, valid_matrix, item_map) -> list:
        """Get validation items for a user (convert from internal ID to real item IDs)"""
        if valid_matrix is None or user_internal_id >= valid_matrix.shape[0]:
            return []

        # Get item indices that this user interacted with in validation set
        valid_item_indices = valid_matrix[user_internal_id].nonzero()[1]

        # Convert to real item IDs
        valid_items = []
        for item_idx in valid_item_indices:
            real_item_id = item_map.get(str(item_idx))
            if real_item_id:
                valid_items.append(real_item_id)

        return valid_items

    def get_user_test_items(self, user_internal_id: int, test_matrix, item_map) -> list:
        """Get test items for a user (convert from internal ID to real item IDs)"""
        if test_matrix is None or user_internal_id >= test_matrix.shape[0]:
            return []

        # Get item indices that this user interacted with in test set
        test_item_indices = test_matrix[user_internal_id].nonzero()[1]

        # Convert to real item IDs
        test_items = []
        for item_idx in test_item_indices:
            real_item_id = item_map.get(str(item_idx))
            if real_item_id:
                test_items.append(real_item_id)

        return test_items

    async def download_validation_items(self, user_internal_id: int, user_real_id: str, valid_matrix, item_map,
                                      download_dir: str, max_duration_hours: float = 0.1,
                                      max_videos_per_playlist: int = 1):
        """Download and process validation items for a user"""
        validation_items = self.get_user_validation_items(user_internal_id, valid_matrix, item_map)

        if not validation_items:
            print(f"⚠️  No validation items found for user {user_real_id}")
            return []

        print(f"📥 Downloading {len(validation_items)} validation items for user {user_real_id}")

        # Create validation folder
        validation_dir = os.path.join(download_dir, str(user_real_id), "validation")
        os.makedirs(validation_dir, exist_ok=True)

        # Download validation items in batches
        successful_downloads = 0
        batch_size = 2

        for batch_idx in range(0, len(validation_items), batch_size):
            batch_items = validation_items[batch_idx:batch_idx + batch_size]
            print(f"[INFO] Processing validation batch {batch_idx//batch_size + 1}/{(len(validation_items) + batch_size - 1)//batch_size} ({len(batch_items)} videos)", flush=True)

            # Add delay between batches (except for the first batch)
            if batch_idx > 0:
                delay = random.uniform(5, 10)  # 5-10 seconds between batches
                print(f"[INFO] Waiting {delay:.1f} seconds before next batch...", flush=True)
                await asyncio.sleep(delay)

            # Create batch tasks
            batch_tasks = []
            for item_id in batch_items:
                item_folder = os.path.join(validation_dir, item_id)
                os.makedirs(item_folder, exist_ok=True)
                task = self.download_bilibili_video(item_id, item_folder,
                                                  max_duration_hours=max_duration_hours,
                                                  max_videos_per_playlist=max_videos_per_playlist)
                batch_tasks.append(task)

            # Execute current batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Count successful downloads
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"[ERROR] Validation download exception: {result}")
                elif result:
                    successful_downloads += 1

            print(f"[SUCCESS] Validation batch completed: {sum(1 for r in batch_results if r is True)}/{len(batch_results)} videos downloaded successfully", flush=True)

        print(f"✅ Downloaded {successful_downloads}/{len(validation_items)} validation items for user {user_real_id}")
        return validation_items

    async def download_test_items(self, user_internal_id: int, user_real_id: str, test_matrix, item_map,
                                  download_dir: str, max_duration_hours: float = 0.1,
                                  max_videos_per_playlist: int = 1):
        """Download and process test items for a user"""
        test_items = self.get_user_test_items(user_internal_id, test_matrix, item_map)

        if not test_items:
            print(f"⚠️  No test items found for user {user_real_id}")
            return []

        print(f"📥 Downloading {len(test_items)} test items for user {user_real_id}")

        # Create test folder
        test_dir = os.path.join(download_dir, str(user_real_id), "test")
        os.makedirs(test_dir, exist_ok=True)

        # Download test items in batches
        successful_downloads = 0
        batch_size = 3

        for batch_idx in range(0, len(test_items), batch_size):
            batch_items = test_items[batch_idx:batch_idx + batch_size]
            print(f"[INFO] Processing test batch {batch_idx//batch_size + 1}/{(len(test_items) + batch_size - 1)//batch_size} ({len(batch_items)} videos)", flush=True)

            # Add delay between batches (except for the first batch)
            if batch_idx > 0:
                delay = random.uniform(5, 10)  # 5-10 seconds between batches
                print(f"[INFO] Waiting {delay:.1f} seconds before next batch...", flush=True)
                await asyncio.sleep(delay)

            # Create batch tasks
            batch_tasks = []
            for item_id in batch_items:
                item_folder = os.path.join(test_dir, item_id)
                os.makedirs(item_folder, exist_ok=True)
                task = self.download_bilibili_video(item_id, item_folder,
                                                  max_duration_hours=max_duration_hours,
                                                  max_videos_per_playlist=max_videos_per_playlist)
                batch_tasks.append(task)

            # Execute current batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Count successful downloads
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"[ERROR] Test download exception: {result}")
                elif result:
                    successful_downloads += 1

            print(f"[SUCCESS] Test batch completed: {sum(1 for r in batch_results if r is True)}/{len(batch_results)} videos downloaded successfully", flush=True)

        print(f"✅ Downloaded {successful_downloads}/{len(test_items)} test items for user {user_real_id}")
        return test_items

    async def bilibili_test(self, max_duration_hours=0.1, max_videos_per_playlist=1, global_concurrent=True, max_concurrent_downloads=2):
        """
        Test bilibili video download with configurable limits and global concurrency

        Args:
            max_duration_hours: Maximum video duration in hours (default: 0.1 hour)
            max_videos_per_playlist: Maximum videos to download from playlists (default: 1)
            global_concurrent: Whether to use global concurrency across all users (default: True)
            max_concurrent_downloads: Maximum concurrent downloads globally (default: 10)
        """
        print("\n" + "="*50)
        print("[INFO] 开始 Bilibili 视频下载测试...")
        print(f"[INFO] 视频时长限制: {max_duration_hours} 小时")
        print(f"[INFO] 播放列表视频数量限制: {max_videos_per_playlist} 个")
        print(f"[INFO] 全局并发模式: {'启用' if global_concurrent else '禁用'}")
        if global_concurrent:
            print(f"[INFO] 最大并发下载数: {max_concurrent_downloads}")
        print("="*50)

        origin_work_dir = os.path.dirname(os.getcwd())
        download_dir = os.path.join(origin_work_dir, "download", "bilibili")
        os.makedirs(download_dir, exist_ok=True)
        print("[INFO] 用户数量：", configs['data']['tst_user_num'])
        random_ids = torch.randint(0, configs['data']['tst_user_num'], (30,))
        # random_ids = torch.tensor([1916])
        print(f"[INFO] 随机用户ID: {random_ids}")

        print("[INFO] 生成推荐预测中...", flush=True)
        sample_preds = self.model.sample_predict(random_ids)
        print(f"[SUCCESS] 推荐预测完成: {sample_preds.shape}", flush=True)

        # 自动开始下载
        import sys
        print(f"\n[INFO] 开始自动下载视频...", flush=True)

        print("[INFO] 开始加载数据文件...")
        with open("datasets/general_cf/bilibili/item_map.json", "r", encoding="utf-8") as f:
            item_map = json.load(f)
        with open("datasets/general_cf/bilibili/user_map.json", "r", encoding="utf-8") as f:
            user_map = json.load(f)

        # Load training matrix to get user historical interactions
        import pickle
        with open("datasets/general_cf/bilibili/train_matrix.pkl", "rb") as f:
            train_mat = pickle.load(f)

        # Load validation matrix for validation items download
        try:
            with open("datasets/general_cf/bilibili/valid_matrix.pkl", "rb") as f:
                valid_mat = pickle.load(f)
            print(f"[INFO] 加载验证矩阵成功: {valid_mat.shape}")
        except Exception as e:
            print(f"[WARNING] 无法加载验证矩阵: {e}")
            valid_mat = None

        # Load test matrix for test items download
        try:
            with open("datasets/general_cf/bilibili/test_matrix.pkl", "rb") as f:
                test_mat = pickle.load(f)
            print(f"[INFO] 加载测试矩阵成功: {test_mat.shape}")
        except Exception as e:
            print(f"[WARNING] 无法加载测试矩阵: {e}")
            test_mat = None

        if global_concurrent:
            # Global concurrent mode: collect all download tasks first
            await self._global_concurrent_download(random_ids, sample_preds, item_map, user_map, train_mat, valid_mat, test_mat,
                                                 download_dir, max_duration_hours, max_videos_per_playlist,
                                                 max_concurrent_downloads)
        else:
            # Original per-user concurrent mode
            await self._per_user_concurrent_download(random_ids, sample_preds, item_map, user_map, train_mat, valid_mat, test_mat,
                                                   download_dir, max_duration_hours, max_videos_per_playlist)

    async def _global_concurrent_download(self, random_ids, sample_preds, item_map, user_map, train_mat, valid_mat, test_mat,
                                        download_dir, max_duration_hours, max_videos_per_playlist,
                                        max_concurrent_downloads):
        """
        Global concurrent download: collect all tasks first, then download concurrently across all users
        """
        print(f"\n[INFO] 🚀 启用全局并发下载模式...")

        # Step 1: Collect video statistics from all users
        user_video_mapping = {}  # Track which videos belong to which user
        total_videos = 0

        for i, user_id in enumerate(random_ids):
            user_id_int = int(user_id.item())
            real_user_id = user_map[str(user_id_int)]
            print(f"[INFO] 📊 统计用户 {user_id_int} (真实ID: {real_user_id}) 的视频...")

            os.makedirs(os.path.join(download_dir, str(real_user_id)), exist_ok=True)

            # Get recommended items
            top_item_indices = sample_preds[i].tolist()
            recommended_bvids = [item_map[str(item_id)] for item_id in top_item_indices]

            # Get user's historical interactions from training matrix
            user_interactions = train_mat[user_id_int].nonzero()[1]
            historical_bvids = [item_map[str(item_id)] for item_id in user_interactions if str(item_id) in item_map]

            user_total = len(recommended_bvids) + len(historical_bvids)
            total_videos += user_total
            print(f"[INFO] 用户 {user_id_int}: 推荐 {len(recommended_bvids)} + 历史 {len(historical_bvids)} = {user_total} 个视频")

            # Track mapping and create folders
            user_video_mapping[real_user_id] = {'recommended': [], 'historical': []}

            for bvid in recommended_bvids:
                folder_path = os.path.join(download_dir, str(real_user_id), "recommended", bvid)
                os.makedirs(folder_path, exist_ok=True)
                user_video_mapping[real_user_id]['recommended'].append(bvid)

            for bvid in historical_bvids:
                folder_path = os.path.join(download_dir, str(real_user_id), "historical", bvid)
                os.makedirs(folder_path, exist_ok=True)
                user_video_mapping[real_user_id]['historical'].append(bvid)

        print(f"\n[INFO] 📈 全局统计完成:")
        print(f"  [INFO] 总用户数: {len(random_ids)}")
        print(f"  [INFO] 总视频数: {total_videos}")

        # Step 2: Execute all tasks with global concurrency control
        print(f"\n[INFO] 🎯 开始全局并发下载 (最大并发: {max_concurrent_downloads})...")

        successful_downloads = 0
        failed_downloads = 0
        user_success_count = {uid: {'recommended': 0, 'historical': 0} for uid in user_video_mapping.keys()}

        # Use semaphore to limit global concurrency
        semaphore = asyncio.Semaphore(max_concurrent_downloads)

        async def limited_download(task_info, task):
            async with semaphore:
                try:
                    result = await task
                    if result:
                        user_success_count[task_info['real_user_id']][task_info['type']] += 1
                        print(f"[SUCCESS] ✅ {task_info['bvid']} 下载成功 ({task_info['real_user_id']}/{task_info['type']})")
                        return True
                    else:
                        print(f"[FAILED] ❌ {task_info['bvid']} 下载失败 ({task_info['real_user_id']}/{task_info['type']})")
                        return False
                except Exception as e:
                    print(f"[ERROR] 下载任务异常 {task_info['bvid']}: {e}")
                    return False

        # Create limited tasks with proper task info
        limited_tasks = []
        task_infos = []

        # Recreate tasks with proper info tracking
        for i, user_id in enumerate(random_ids):
            user_id_int = int(user_id.item())
            real_user_id = user_map[str(user_id_int)]

            # Get recommended items
            top_item_indices = sample_preds[i].tolist()
            recommended_bvids = [item_map[str(item_id)] for item_id in top_item_indices]

            # Get user's historical interactions from training matrix
            user_interactions = train_mat[user_id_int].nonzero()[1]
            historical_bvids = [item_map[str(item_id)] for item_id in user_interactions if str(item_id) in item_map]

            # Create tasks for recommended videos
            for bvid in recommended_bvids:
                folder_path = os.path.join(download_dir, str(real_user_id), "recommended", bvid)
                task_info = {
                    'bvid': bvid,
                    'folder_path': folder_path,
                    'user_id': user_id_int,
                    'real_user_id': real_user_id,
                    'type': 'recommended'
                }
                task = self.download_bilibili_video(bvid, folder_path,
                                                  max_duration_hours=max_duration_hours,
                                                  max_videos_per_playlist=max_videos_per_playlist)
                limited_task = limited_download(task_info, task)
                limited_tasks.append(limited_task)
                task_infos.append(task_info)

            # Create tasks for historical videos
            for bvid in historical_bvids:
                folder_path = os.path.join(download_dir, str(real_user_id), "historical", bvid)
                task_info = {
                    'bvid': bvid,
                    'folder_path': folder_path,
                    'user_id': user_id_int,
                    'real_user_id': real_user_id,
                    'type': 'historical'
                }
                task = self.download_bilibili_video(bvid, folder_path,
                                                  max_duration_hours=max_duration_hours,
                                                  max_videos_per_playlist=max_videos_per_playlist)
                limited_task = limited_download(task_info, task)
                limited_tasks.append(limited_task)
                task_infos.append(task_info)

        # Execute all downloads concurrently
        start_time = time.time()
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        end_time = time.time()

        # Count results
        for result in results:
            if isinstance(result, Exception):
                failed_downloads += 1
                print(f"[ERROR] 下载异常: {result}")
            elif result:
                successful_downloads += 1
            else:
                failed_downloads += 1

        # Print summary
        print(f"\n[SUMMARY] 🎉 全局并发下载完成:")
        print(f"  [SUCCESS] 成功下载: {successful_downloads}/{len(limited_tasks)} 个视频")
        print(f"  [ERROR] 失败下载: {failed_downloads}/{len(limited_tasks)} 个视频")
        print(f"  [INFO] 总耗时: {end_time - start_time:.2f} 秒")
        if end_time - start_time > 0:
            print(f"  [INFO] 平均速度: {len(limited_tasks)/(end_time - start_time):.2f} 任务/秒")

        # Print per-user summary
        for real_user_id, counts in user_success_count.items():
            total_user_videos = len(user_video_mapping[real_user_id]['recommended']) + len(user_video_mapping[real_user_id]['historical'])
            total_user_success = counts['recommended'] + counts['historical']
            print(f"  [USER] {real_user_id}: {total_user_success}/{total_user_videos} 成功 (推荐: {counts['recommended']}, 历史: {counts['historical']})")

        print("="*50, flush=True)

        # Step 3: Download validation items for each user
        if valid_mat is not None:
            print(f"\n[INFO] 📥 开始下载 validation 数据...")
            validation_start_time = time.time()

            for i, user_id in enumerate(random_ids):
                user_id_int = int(user_id.item())
                real_user_id = user_map[str(user_id_int)]

                print(f"[INFO] 下载用户 {user_id_int} (真实ID: {real_user_id}) 的 validation 数据...")
                await self.download_validation_items(
                    user_id_int, real_user_id, valid_mat, item_map, download_dir,
                    max_duration_hours, max_videos_per_playlist
                )

            validation_end_time = time.time()
            print(f"\n[INFO] ✅ Validation 数据下载完成，耗时: {validation_end_time - validation_start_time:.2f} 秒")
        else:
            print(f"\n[WARNING] ⚠️  跳过 validation 数据下载 (未找到验证矩阵)")

        # Step 4: Download test items for each user
        if test_mat is not None:
            print(f"\n[INFO] 📥 开始下载 test 数据...")
            test_start_time = time.time()

            for i, user_id in enumerate(random_ids):
                user_id_int = int(user_id.item())
                real_user_id = user_map[str(user_id_int)]

                print(f"[INFO] 下载用户 {user_id_int} (真实ID: {real_user_id}) 的 test 数据...")
                await self.download_test_items(
                    user_id_int, real_user_id, test_mat, item_map, download_dir,
                    max_duration_hours, max_videos_per_playlist
                )

            test_end_time = time.time()
            print(f"\n[INFO] ✅ Test 数据下载完成，耗时: {test_end_time - test_start_time:.2f} 秒")
        else:
            print(f"\n[WARNING] ⚠️  跳过 test 数据下载 (未找到测试矩阵)")

        print("="*50, flush=True)


    async def _per_user_concurrent_download(self, random_ids, sample_preds, item_map, user_map, train_mat, valid_mat, test_mat,
                                          download_dir, max_duration_hours, max_videos_per_playlist):
        """
        Original per-user concurrent download mode
        """
        print(f"\n[INFO] 📱 使用原始的按用户并发下载模式...")

        for i, user_id in enumerate(random_ids):
            user_id_int = int(user_id.item())
            real_user_id = user_map[str(user_id_int)]
            print("[INFO] 用户ID: ", user_id_int, "[INFO] 真实用户ID: ", real_user_id)
            os.makedirs(os.path.join(download_dir, str(real_user_id)), exist_ok=True)

            # Get recommended items
            top_item_indices = sample_preds[i].tolist()
            recommended_bvids = [item_map[str(item_id)] for item_id in top_item_indices]
            print(f"[INFO] 用户 {user_id_int} 推荐视频: {recommended_bvids}", flush=True)

            # Get user's historical interactions from training matrix
            user_interactions = train_mat[user_id_int].nonzero()[1]  # Get column indices (item indices)
            historical_bvids = [item_map[str(item_id)] for item_id in user_interactions if str(item_id) in item_map]
            print(f"[INFO] 用户 {user_id_int} 历史视频 ({len(historical_bvids)} 个): {historical_bvids[:10]}...", flush=True)  # Show first 10

            # Combine recommended and historical items for download
            all_bvids = recommended_bvids + historical_bvids
            print(f"[INFO] 用户 {user_id_int} 总计下载: {len(all_bvids)} 个视频 (推荐: {len(recommended_bvids)}, 历史: {len(historical_bvids)})", flush=True)

            tasks = []
            # Create separate folders for recommended and historical items
            for bvid in recommended_bvids:
                folder_path = os.path.join(download_dir, str(real_user_id), "recommended", bvid)
                os.makedirs(folder_path, exist_ok=True)
                task = self.download_bilibili_video(bvid, folder_path,
                                                  max_duration_hours=max_duration_hours,
                                                  max_videos_per_playlist=max_videos_per_playlist)
                tasks.append(task)

            for bvid in historical_bvids:
                folder_path = os.path.join(download_dir, str(real_user_id), "historical", bvid)
                os.makedirs(folder_path, exist_ok=True)
                task = self.download_bilibili_video(bvid, folder_path,
                                                  max_duration_hours=max_duration_hours,
                                                  max_videos_per_playlist=max_videos_per_playlist)
                tasks.append(task)

            # Execute downloads in batches of 3 to balance speed and avoid rate limiting
            successful_downloads = 0
            batch_size = 3

            for batch_idx in range(0, len(tasks), batch_size):
                batch_tasks = tasks[batch_idx:batch_idx + batch_size]
                print(f"[INFO] 处理批次 {batch_idx//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size} ({len(batch_tasks)} 个视频)", flush=True)

                # Add delay between batches (except for the first batch)
                if batch_idx > 0:
                    delay = random.uniform(2, 3)  # 5-10 seconds between batches
                    print(f"[INFO] 等待 {delay:.1f} 秒后开始下一批次...", flush=True)
                    await asyncio.sleep(delay)

                # Execute current batch concurrently
                batch_results = await asyncio.gather(*batch_tasks)
                successful_downloads += sum(batch_results)

                print(f"[SUCCESS] 批次完成: {sum(batch_results)}/{len(batch_results)} 个视频下载成功", flush=True)

            print(f"\n[SUMMARY] 用户 {user_id_int} 下载总结:")
            print(f"  [SUCCESS] 成功下载: {successful_downloads}/{len(tasks)} 个视频")
            print(f"  [INFO] 推荐视频: {len(recommended_bvids)} 个")
            print(f"  [INFO] 历史视频: {len(historical_bvids)} 个")

            # Download validation items for this user
            if valid_mat is not None:
                print(f"  [INFO] 📥 开始下载用户 {user_id_int} 的 validation 数据...")
                await self.download_validation_items(
                    user_id_int, real_user_id, valid_mat, item_map, download_dir,
                    max_duration_hours, max_videos_per_playlist
                )
            else:
                print(f"  [WARNING] ⚠️  跳过 validation 数据下载 (未找到验证矩阵)")

            # Download test items for this user
            if test_mat is not None:
                print(f"  [INFO] 📥 开始下载用户 {user_id_int} 的 test 数据...")
                await self.download_test_items(
                    user_id_int, real_user_id, test_mat, item_map, download_dir,
                    max_duration_hours, max_videos_per_playlist
                )
            else:
                print(f"  [WARNING] ⚠️  跳过 test 数据下载 (未找到测试矩阵)")

            print("="*50, flush=True)


class DoubanDownloader:
    """
    Douban content downloader for text and images

    Usage example:
        downloader = DoubanDownloader(model)
        await downloader.douban_test()
    """
    def __init__(self, model):
        self.model = model

    def get_user_validation_items(self, user_internal_id: int, valid_matrix, item_map) -> list:
        """Get validation items for a user (convert from internal ID to real item IDs)"""
        if valid_matrix is None or user_internal_id >= valid_matrix.shape[0]:
            return []

        # Get item indices that this user interacted with in validation set
        valid_item_indices = valid_matrix[user_internal_id].nonzero()[1]

        # Convert to real item IDs
        valid_items = []
        for item_idx in valid_item_indices:
            real_item_id = item_map.get(str(item_idx))
            if real_item_id:
                valid_items.append(real_item_id)

        return valid_items

    def get_user_test_items(self, user_internal_id: int, test_matrix, item_map) -> list:
        """Get test items for a user (convert from internal ID to real item IDs)"""
        if test_matrix is None or user_internal_id >= test_matrix.shape[0]:
            return []

        # Get item indices that this user interacted with in test set
        test_item_indices = test_matrix[user_internal_id].nonzero()[1]

        # Convert to real item IDs
        test_items = []
        for item_idx in test_item_indices:
            real_item_id = item_map.get(str(item_idx))
            if real_item_id:
                test_items.append(real_item_id)

        return test_items

    async def download_validation_items(self, user_internal_id: int, user_real_id: str, valid_matrix, item_map,
                                      download_dir: str, content_mapping: dict):
        """Download and process validation items for a user"""
        validation_items = self.get_user_validation_items(user_internal_id, valid_matrix, item_map)

        if not validation_items:
            print(f"⚠️  No validation items found for user {user_real_id}")
            return []

        print(f"📥 Processing {len(validation_items)} validation items for user {user_real_id}")

        # Create validation folder
        validation_dir = os.path.join(download_dir, str(user_real_id), "validation")
        os.makedirs(validation_dir, exist_ok=True)

        # Process validation items
        successful_downloads = 0

        for douban_id in validation_items:
            # Use douban_id to lookup in content mapping
            matching_content = content_mapping.get(douban_id)

            if matching_content:
                try:
                    success = await self.process_douban_content(
                        matching_content,
                        validation_dir,
                        "validation",
                        douban_id
                    )
                    if success:
                        successful_downloads += 1

                    # Add small delay to avoid overwhelming the system
                    await asyncio.sleep(0.2)

                except Exception as e:
                    print(f"[ERROR] 处理validation内容时出错 {douban_id}: {e}")
            else:
                print(f"[WARNING] 未找到validation内容 '{douban_id}' 的对应数据")

        print(f"✅ Processed {successful_downloads}/{len(validation_items)} validation items for user {user_real_id}")
        return validation_items

    async def download_test_items(self, user_internal_id: int, user_real_id: str, test_matrix, item_map,
                                  download_dir: str, content_mapping: dict):
        """Download and process test items for a user"""
        test_items = self.get_user_test_items(user_internal_id, test_matrix, item_map)

        if not test_items:
            print(f"⚠️  No test items found for user {user_real_id}")
            return []

        print(f"📥 Processing {len(test_items)} test items for user {user_real_id}")

        # Create test folder
        test_dir = os.path.join(download_dir, str(user_real_id), "test")
        os.makedirs(test_dir, exist_ok=True)

        # Process test items
        successful_downloads = 0

        for douban_id in test_items:
            # Use douban_id to lookup in content mapping
            matching_content = content_mapping.get(douban_id)

            if matching_content:
                try:
                    success = await self.process_douban_content(
                        matching_content,
                        test_dir,
                        "test",
                        douban_id
                    )
                    if success:
                        successful_downloads += 1

                    # Add small delay to avoid overwhelming the system
                    await asyncio.sleep(0.2)

                except Exception as e:
                    print(f"[ERROR] 处理test内容时出错 {douban_id}: {e}")
            else:
                print(f"[WARNING] 未找到test内容 '{douban_id}' 的对应数据")

        print(f"✅ Processed {successful_downloads}/{len(test_items)} test items for user {user_real_id}")
        return test_items

    async def download_douban_images(self, images_str, save_path, max_retries=3, max_images=3):
        """Download images from semicolon-separated URLs with concurrent processing

        Args:
            images_str: Semicolon-separated image URLs
            save_path: Path to save the images (directly in this folder)
            max_retries: Maximum retry attempts
            max_images: Maximum number of images to download per item (default: 5)
        """
        if not images_str or images_str.strip() == '':
            return True

        # Split images by semicolon and filter out empty strings
        image_urls = [url.strip() for url in images_str.split(';') if url.strip()]

        if not image_urls:
            return True

        # Limit the number of images to download
        if len(image_urls) > max_images:
            print(f"[INFO] 图片数量限制: 从 {len(image_urls)} 张中选择前 {max_images} 张下载")
            image_urls = image_urls[:max_images]

        print(f"[INFO] 并发下载 {len(image_urls)} 张图片到 {save_path}")

        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Create concurrent download tasks
        download_tasks = []
        for i, image_url in enumerate(image_urls):
            task = self.download_single_image(image_url, save_path, i+1, max_retries)
            download_tasks.append(task)

        # Limit concurrent downloads to avoid overwhelming the server
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent image downloads

        async def limited_download(task):
            async with semaphore:
                return await task

        limited_tasks = [limited_download(task) for task in download_tasks]

        # Execute all downloads concurrently
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)

        # Count successful downloads
        successful_downloads = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[ERROR] 图片 {i+1} 下载异常: {result}")
            elif result:
                successful_downloads += 1

        print(f"[SUCCESS] 图片下载完成: {successful_downloads}/{len(image_urls)} 张成功")
        return successful_downloads > 0

    async def download_single_image(self, image_url, save_path, image_index, max_retries=3):
        """Download a single image with retry mechanism

        Args:
            image_url: URL of the image to download
            save_path: Directory to save the image
            image_index: Index of the image (for filename)
            max_retries: Maximum retry attempts

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid being detected as bot
                if attempt > 0:
                    delay = random.uniform(1, 3)  # 1-3 seconds delay
                    print(f"[INFO] 重试下载图片 {image_index}, 等待 {delay:.1f} 秒...")
                    await asyncio.sleep(delay)

                # Extract file extension from URL
                parsed_url = image_url.split('?')[0]  # Remove query parameters
                file_extension = os.path.splitext(parsed_url)[1]
                if not file_extension:
                    file_extension = '.jpg'  # Default to jpg if no extension

                # Generate filename
                filename = f"image_{image_index:03d}{file_extension}"
                filepath = os.path.join(save_path, filename)

                # Download image using curl (similar to yt-dlp approach)
                cmd = [
                    "curl",
                    "-L",  # Follow redirects
                    "-o", filepath,
                    "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "--referer", "https://www.douban.com/",
                    "--connect-timeout", "30",
                    "--max-time", "60",
                    image_url
                ]

                # Run curl command asynchronously
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await process.communicate()

                if process.returncode == 0 and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    print(f"[SUCCESS] 图片下载成功: {filename}")
                    return True
                else:
                    try:
                        error_msg = stderr.decode('utf-8')
                    except UnicodeDecodeError:
                        error_msg = stderr.decode('gbk', errors='ignore')

                    print(f"[WARNING] 图片下载失败 {filename}: {error_msg}")

                    # Remove empty file if exists
                    if os.path.exists(filepath):
                        os.remove(filepath)

                    if attempt == max_retries - 1:
                        print(f"[ERROR] 图片 {filename} 所有重试都失败")

            except Exception as e:
                print(f"[ERROR] 下载图片异常 {filename}: {e}")
                if attempt == max_retries - 1:
                    return False

        return False

    def save_text_content(self, title, content, save_path):
        """Save title and content as txt file

        Args:
            title: Title to use as filename
            content: Content to save (will be truncated to first 2000 characters)
            save_path: Directory path to save the file
        """
        try:
            # Clean title for filename (remove invalid characters)
            import re
            clean_title = re.sub(r'[<>:"/\\|?*]', '_', title)
            clean_title = clean_title.strip()

            # Limit filename length
            if len(clean_title) > 100:
                clean_title = clean_title[:100]

            if not clean_title:
                clean_title = "untitled"

            filename = f"{clean_title}.txt"
            filepath = os.path.join(save_path, filename)

            # Ensure directory exists
            os.makedirs(save_path, exist_ok=True)

            # Limit content to first 2000 characters
            if len(content) > 2000:
                content = content[:2000]
                print(f"[INFO] 内容已截断至前2000字符: {filename}")

            # Write content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"[SUCCESS] 文本保存成功: {filename}")
            return True

        except Exception as e:
            print(f"[ERROR] 保存文本文件失败: {e}")
            return False

    def clean_folder_name(self, title):
        """Clean title for use as folder name

        Args:
            title: Title to clean

        Returns:
            Clean folder name
        """
        import re
        # Remove invalid characters for folder names
        clean_title = re.sub(r'[<>:"/\\|?*]', '_', title)
        clean_title = clean_title.strip()

        # Limit folder name length
        if len(clean_title) > 50:
            clean_title = clean_title[:50]

        if not clean_title:
            clean_title = "untitled"

        return clean_title

    async def process_douban_content(self, content_data, folder_path, content_type, douban_id=None):
        """Process douban content (text and images)

        Args:
            content_data: Dictionary containing title, content, images
            folder_path: Path to save the content
            content_type: "recommended" or "historical"
            douban_id: Douban ID to use as folder name, if None use title
        """
        title = content_data.get('title', 'untitled')
        content = content_data.get('content', '')
        images = content_data.get('images', '')

        # Create folder with douban_id or clean title name
        if douban_id:
            folder_name = str(douban_id)
        else:
            folder_name = self.clean_folder_name(title)

        item_folder = os.path.join(folder_path, folder_name)
        os.makedirs(item_folder, exist_ok=True)

        # Save text content (still use title for filename)
        text_success = self.save_text_content(title, content, item_folder)

        # Download images
        image_success = await self.download_douban_images(images, item_folder, max_images=3)

        if text_success:
            print(f"[SUCCESS] {content_type} 内容处理完成: {folder_name}")
            return True
        else:
            print(f"[ERROR] {content_type} 内容处理失败: {folder_name}")
            return False

    async def process_content_batch(self, content_mapping, item_ids, folder_path, content_type):
        """Process a batch of content items concurrently

        Args:
            content_mapping: Dictionary mapping item_id to content data
            item_ids: List of douban IDs to process
            folder_path: Base folder path for this content type
            content_type: "推荐" or "历史"

        Returns:
            Number of successfully processed items
        """
        print(f"[INFO] 开始并发处理{content_type}内容 ({len(item_ids)} 项)...")

        # Create tasks for concurrent processing within this batch
        batch_tasks = []
        for i, douban_id in enumerate(item_ids):
            # Use douban_id to lookup in content mapping
            matching_content = content_mapping.get(douban_id)

            if matching_content:
                task = self.process_single_content(
                    matching_content, folder_path, content_type, i+1, len(item_ids), douban_id
                )
                batch_tasks.append(task)
            else:
                print(f"[WARNING] 未找到{content_type}内容 '{douban_id}' 的对应数据")

        if not batch_tasks:
            print(f"[WARNING] {content_type}内容批次无有效任务")
            return 0

        # Process batch concurrently with limited concurrency to avoid overwhelming the system
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent downloads per batch

        async def limited_process(task):
            async with semaphore:
                return await task

        limited_tasks = [limited_process(task) for task in batch_tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)

        # Count successful downloads
        successful_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[ERROR] {content_type}内容 {i+1} 处理失败: {result}")
            elif result:
                successful_count += 1

        print(f"[SUCCESS] {content_type}内容批次完成: {successful_count}/{len(batch_tasks)} 个成功")
        return successful_count

    async def process_single_content(self, content_data, folder_path, content_type, index, total, douban_id=None):
        """Process a single content item

        Args:
            content_data: Content data dictionary
            folder_path: Folder path for this content type
            content_type: "推荐" or "历史"
            index: Current item index
            total: Total items in batch
            douban_id: Douban ID to use as folder name

        Returns:
            True if successful, False otherwise
        """
        title = content_data['title'][:30] + "..." if len(content_data['title']) > 30 else content_data['title']
        display_name = douban_id if douban_id else title
        print(f"[INFO] 处理{content_type}内容 {index}/{total}: {display_name}")

        try:
            success = await self.process_douban_content(
                content_data,
                folder_path,
                content_type,
                douban_id
            )

            # Add small delay to avoid overwhelming the server
            await asyncio.sleep(0.3)

            return success

        except Exception as e:
            print(f"[ERROR] 处理{content_type}内容时出错: {e}")
            return False

    async def douban_test(self):
        """
        Test douban content download with concurrent processing
        """
        print("\n" + "="*50)
        print("[INFO] 开始 Douban 内容下载测试 (并发模式)...")
        print("="*50)

        origin_work_dir = os.path.dirname(os.getcwd())
        download_dir = os.path.join(origin_work_dir, "download", "douban")
        os.makedirs(download_dir, exist_ok=True)

        print("[INFO] 用户数量：", configs['data']['tst_user_num'])
        random_ids = torch.randint(0, configs['data']['tst_user_num'], (30,))
        print(f"[INFO] 随机用户ID: {random_ids}")

        print("[INFO] 生成推荐预测中...", flush=True)
        sample_preds = self.model.sample_predict(random_ids)
        print(f"[SUCCESS] 推荐预测完成: {sample_preds.shape}", flush=True)

        # 自动开始下载
        print(f"\n[INFO] 开始自动下载内容...", flush=True)

        print("[INFO] 开始加载数据文件...")
        with open("datasets/general_cf/douban/item_map.json", "r", encoding="utf-8") as f:
            item_map = json.load(f)
        with open("datasets/general_cf/douban/user_map.json", "r", encoding="utf-8") as f:
            user_map = json.load(f)

        # Load training matrix to get user historical interactions
        import pickle
        with open("datasets/general_cf/douban/train_matrix.pkl", "rb") as f:
            train_mat = pickle.load(f)

        # Load validation matrix for validation items download
        try:
            with open("datasets/general_cf/douban/valid_matrix.pkl", "rb") as f:
                valid_mat = pickle.load(f)
            print(f"[INFO] 加载验证矩阵成功: {valid_mat.shape}")
        except Exception as e:
            print(f"[WARNING] 无法加载验证矩阵: {e}")
            valid_mat = None

        # Load test matrix for test items download
        try:
            with open("datasets/general_cf/douban/test_matrix.pkl", "rb") as f:
                test_mat = pickle.load(f)
            print(f"[INFO] 加载测试矩阵成功: {test_mat.shape}")
        except Exception as e:
            print(f"[WARNING] 无法加载测试矩阵: {e}")
            test_mat = None

        for i, user_id in enumerate(random_ids):
            user_id_int = int(user_id.item())
            real_user_id = user_map[str(user_id_int)]
            print("[INFO] 用户ID: ", user_id_int, "[INFO] 真实用户ID: ", real_user_id)

            user_download_dir = os.path.join(download_dir, str(real_user_id))
            os.makedirs(user_download_dir, exist_ok=True)

            # Get recommended items (item_map contains douban IDs)
            top_item_indices = sample_preds[i].tolist()
            recommended_douban_ids = [item_map[str(item_id)] for item_id in top_item_indices if str(item_id) in item_map]
            print(f"[INFO] 用户 {user_id_int} 推荐内容: {recommended_douban_ids}", flush=True)

            # Get user's historical interactions from training matrix
            # Convert coo_matrix to csr_matrix for indexing
            train_mat_csr = train_mat.tocsr()
            user_interactions = train_mat_csr[user_id_int].nonzero()[1]  # Get column indices (item indices)
            historical_douban_ids = [item_map[str(item_id)] for item_id in user_interactions if str(item_id) in item_map]
            print(f"[INFO] 用户 {user_id_int} 历史内容 ({len(historical_douban_ids)} 个): {historical_douban_ids[:10]}...", flush=True)  # Show first 10

            # Combine recommended and historical items for download
            all_douban_ids = recommended_douban_ids + historical_douban_ids
            print(f"[INFO] 用户 {user_id_int} 总计下载: {len(all_douban_ids)} 个内容 (推荐: {len(recommended_douban_ids)}, 历史: {len(historical_douban_ids)})", flush=True)

            # Load douban data to get actual content
            douban_data_dir = os.path.join(origin_work_dir, "dataset", "douban", str(real_user_id))

            if not os.path.exists(douban_data_dir):
                print(f"[WARNING] 用户 {real_user_id} 的数据目录不存在: {douban_data_dir}")
                continue

            # Create separate folders for recommended and historical items
            recommended_folder = os.path.join(user_download_dir, "recommended")
            historical_folder = os.path.join(user_download_dir, "historical")
            os.makedirs(recommended_folder, exist_ok=True)
            os.makedirs(historical_folder, exist_ok=True)

            # Process CSV files in user directory to build content mapping for historical items
            csv_files = [f for f in os.listdir(douban_data_dir) if f.endswith('.csv')]

            # Build a mapping from douban ID to content data
            content_mapping = {}

            # First, load content from user's CSV files (historical items)
            for csv_file in csv_files:
                csv_path = os.path.join(douban_data_dir, csv_file)

                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)

                    print(f"[INFO] 加载用户CSV文件: {csv_file}, 包含 {len(df)} 条记录")

                    for idx, row in df.iterrows():
                        title = str(row.get('title', f'untitled_{idx}'))
                        douban_id = str(row.get('doubanID', ''))

                        if douban_id:  # Only add if douban_id exists
                            content_data = {
                                'title': title,
                                'content': str(row.get('content', '')),
                                'images': str(row.get('images', ''))
                            }

                            # Use douban ID as the key for mapping
                            content_mapping[douban_id] = content_data

                except Exception as e:
                    print(f"[ERROR] 加载用户CSV文件 {csv_file} 时出错: {e}")
                    continue

            # Second, load content from enhanced douban_mapping.json for recommended items
            print(f"[INFO] 从douban_mapping.json 加载推荐内容...")

            # Find recommended items that are not in user's CSV data
            missing_recommended_ids = [douban_id for douban_id in recommended_douban_ids
                                     if douban_id not in content_mapping]

            if missing_recommended_ids:
                print(f"[INFO] 需要从 douban_mapping.json 加载 {len(missing_recommended_ids)} 个推荐内容")

                # Load enhanced mapping file
                douban_mapping_file = os.path.join(origin_work_dir, "douban_mapping.json")
                if os.path.exists(douban_mapping_file):
                    try:
                        with open(douban_mapping_file, 'r', encoding='utf-8') as f:
                            douban_mapping = json.load(f)

                        loaded_count = 0
                        for douban_id in missing_recommended_ids:
                            # For douban, the key is the douban_id without .json extension

                            if douban_id in douban_mapping:
                                item_data = douban_mapping[douban_id]
                                if isinstance(item_data, dict):
                                    # New enhanced format
                                    content_data = {
                                        'title': item_data.get('title', f'untitled_{douban_id}'),
                                        'content': item_data.get('content', ''),
                                        'images': item_data.get('images', ''),
                                        'videos': item_data.get('videos', '')
                                    }
                                else:
                                    # Old format (just title)
                                    content_data = {
                                        'title': item_data,
                                        'content': '',
                                        'images': '',
                                        'videos': ''
                                    }

                                content_mapping[douban_id] = content_data
                                loaded_count += 1
                                print(f"[INFO] 从 mapping 加载推荐内容: {douban_id} - {content_data['title'][:50]}...")
                            else:
                                print(f"[WARNING] 推荐内容 {douban_id} 在 mapping 中未找到")

                        print(f"[SUCCESS] 从 douban_mapping.json 成功加载 {loaded_count}/{len(missing_recommended_ids)} 个推荐内容")

                    except Exception as e:
                        print(f"[ERROR] 加载 douban_mapping.json 时出错: {e}")
                else:
                    print(f"[WARNING] douban_mapping.json 文件不存在: {douban_mapping_file}")
            else:
                print(f"[INFO] 所有推荐内容都已在用户历史数据中，无需额外加载")

            print(f"[INFO] 内容映射构建完成，总计 {len(content_mapping)} 个内容")

            # Create concurrent tasks for recommended and historical content
            tasks = []
            start_time = time.time()

            # Create task for recommended content processing
            if recommended_douban_ids:
                recommended_task = self.process_content_batch(
                    content_mapping, recommended_douban_ids, recommended_folder, "推荐"
                )
                tasks.append(recommended_task)

            # Create task for historical content processing
            if historical_douban_ids:
                historical_task = self.process_content_batch(
                    content_mapping, historical_douban_ids, historical_folder, "历史"
                )
                tasks.append(historical_task)

            print(f"\n[INFO] 开始并发处理 {len(tasks)} 个任务...")

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            total_time = end_time - start_time

            # Process results
            total_successful = 0
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"[ERROR] 任务 {j+1} 执行失败: {result}")
                else:
                    total_successful += result

            print(f"\n[SUMMARY] 用户 {user_id_int} 下载总结:")
            print(f"  [SUCCESS] 成功处理: {total_successful} 个内容")
            print(f"  [INFO] 推荐内容: {len(recommended_douban_ids)} 个")
            print(f"  [INFO] 历史内容: {len(historical_douban_ids)} 个")
            print(f"  [INFO] 总耗时: {total_time:.2f} 秒")
            print(f"  [INFO] CSV文件数: {len(csv_files)} 个")

            # Download validation items for this user
            if valid_mat is not None:
                print(f"  [INFO] 📥 开始处理用户 {user_id_int} 的 validation 数据...")
                await self.download_validation_items(
                    user_id_int, real_user_id, valid_mat, item_map, download_dir, content_mapping
                )
            else:
                print(f"  [WARNING] ⚠️  跳过 validation 数据处理 (未找到验证矩阵)")

            # Download test items for this user
            if test_mat is not None:
                print(f"  [INFO] 📥 开始处理用户 {user_id_int} 的 test 数据...")
                await self.download_test_items(
                    user_id_int, real_user_id, test_mat, item_map, download_dir, content_mapping
                )
            else:
                print(f"  [WARNING] ⚠️  跳过 test 数据处理 (未找到测试矩阵)")

            print("="*50, flush=True)


class RedbookDownloader:
    """
    Redbook content downloader for videos and image notes

    Usage example:
        downloader = RedbookDownloader(model)
        await downloader.redbook_test()
    """
    def __init__(self, model):
        self.model = model
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.xiaohongshu.com/',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        self.session.headers.update(self.headers)
        self.timeout = 30
        self.max_retries = 3

        # Download limits
        self.max_images_per_note = 2  # Maximum images per note
        self.max_videos_per_note = 1  # Maximum videos per note (usually 1 anyway)

        # Request statistics
        self.successful_requests = 0
        self.failed_requests = 0

    def add_cookies(self, cookie_str: str):
        """Add cookies to session"""
        if not cookie_str:
            return

        cookies = {}
        for item in cookie_str.split(';'):
            key_val = item.strip().split('=', 1)
            if len(key_val) == 2:
                cookies[key_val[0]] = key_val[1]

        self.session.cookies.update(cookies)
        print(f"✅ Added {len(cookies)} cookies")

    def set_download_limits(self, max_images_per_note=5, max_videos_per_note=1):
        """Set download limits for images and videos

        Args:
            max_images_per_note: Maximum number of images to download per note (default: 5)
            max_videos_per_note: Maximum number of videos to download per note (default: 1)
        """
        self.max_images_per_note = max_images_per_note
        self.max_videos_per_note = max_videos_per_note
        print(f"✅ Set download limits: max_images={max_images_per_note}, max_videos={max_videos_per_note}")

    def get_user_validation_items(self, user_internal_id: int, valid_matrix, item_map) -> list:
        """Get validation items for a user (convert from internal ID to real item IDs)"""
        if valid_matrix is None or user_internal_id >= valid_matrix.shape[0]:
            return []

        # Get item indices that this user interacted with in validation set
        valid_item_indices = valid_matrix[user_internal_id].nonzero()[1]

        # Convert to real item IDs
        valid_items = []
        for item_idx in valid_item_indices:
            real_item_id = item_map.get(str(item_idx))
            if real_item_id:
                valid_items.append(real_item_id)

        return valid_items

    def get_user_test_items(self, user_internal_id: int, test_matrix, item_map) -> list:
        """Get test items for a user (convert from internal ID to real item IDs)"""
        if test_matrix is None or user_internal_id >= test_matrix.shape[0]:
            return []

        # Get item indices that this user interacted with in test set
        test_item_indices = test_matrix[user_internal_id].nonzero()[1]

        # Convert to real item IDs
        test_items = []
        for item_idx in test_item_indices:
            real_item_id = item_map.get(str(item_idx))
            if real_item_id:
                test_items.append(real_item_id)

        return test_items

    def _sanitize_filename(self, filename: str) -> str:
        """Clean filename of invalid characters"""
        if not filename or filename.strip() == '':
            return "无标题"

        filename = re.sub(r'[\uFEFF\u200B-\u200D\uFFFC]', '', filename)
        invalid_chars = r'<>:"/\\|?*#\r\n\t'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        filename = re.sub(r'_+', '_', filename)
        filename = filename.strip('_').strip()
        filename = re.sub(r'\.{2,}', '_', filename)

        return filename[:100] if filename else "无标题"

    def _rotate_user_agent(self):
        """Rotate User-Agent to avoid detection"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/140.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
        ]
        self.session.headers['User-Agent'] = random.choice(user_agents)

    def _make_request(self, url: str, retry_count: int = 0) -> Optional[str]:
        """Make HTTP request with retry mechanism"""
        if retry_count >= self.max_retries:
            print(f"        ✗✗ Request failed after max retries: {url}")
            self.failed_requests += 1
            return None

        time.sleep(random.uniform(1.5, 2.5))
        self._rotate_user_agent()

        try:
            if not url.startswith('http'):
                url = 'https://www.xiaohongshu.com' + url

            print(f"        → Request: {url[:100]}...")
            resp = self.session.get(url, timeout=self.timeout)

            if resp.status_code == 403:
                print(f"        ⚠⚠⚠ Blocked by server (403), possible anti-bot")
                time.sleep(5 * 2)
                return self._make_request(url, retry_count + 1)
            elif resp.status_code == 429:
                print(f"        ⚠⚠⚠ Too many requests (429), waiting and retry")
                time.sleep(5 * (retry_count + 1))
                return self._make_request(url, retry_count + 1)
            elif resp.status_code >= 500:
                print(f"        ⚠⚠⚠ Server error ({resp.status_code}), waiting and retry")
                time.sleep(5)
                return self._make_request(url, retry_count + 1)

            resp.raise_for_status()

            if len(resp.text) < 1000 and "验证" in resp.text:
                print(f"        ⚠⚠⚠ Possible verification page encountered")
                time.sleep(5 * 2)
                return self._make_request(url, retry_count + 1)

            self.successful_requests += 1
            return resp.text

        except requests.exceptions.RequestException as e:
            print(f"        ✗✗ Request failed ({retry_count + 1}/{self.max_retries}): {e}")
            time.sleep(5 * (retry_count + 1))
            return self._make_request(url, retry_count + 1)
        except Exception as e:
            print(f"        ✗✗ Request exception: {e}")
            self.failed_requests += 1
            return None

    def _normalize_url(self, url: str) -> Optional[str]:
        """Normalize URL"""
        if not url:
            return None

        invalid_patterns = ['fe-platform', 'avatar', 'icon', 'logo', 'default']
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in invalid_patterns):
            return None

        if url.startswith("//"):
            return "https:" + url
        elif url.startswith("/"):
            return "https://www.xiaohongshu.com" + url

        return url

    def _get_file_extension(self, url: str) -> str:
        """Get file extension from URL"""
        if not url:
            return '.jpg'

        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path

        if '.' in path:
            ext = '.' + path.split('.')[-1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.mp4', '.avi', '.mov']:
                return ext

        return '.jpg'

    async def download_file(self, url: str, save_path: str) -> bool:
        """Download file"""
        await asyncio.sleep(random.uniform(1.5, 2.5))
        try:
            headers = {
                'User-Agent': self.session.headers['User-Agent'],
                'Referer': 'https://www.xiaohongshu.com/',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
            }

            resp = requests.get(url, headers=headers, stream=True, timeout=self.timeout)
            resp.raise_for_status()

            content_type = resp.headers.get('content-type', '')
            if not content_type.startswith('image/') and not content_type.startswith('video/'):
                print(f"        ⚠⚠⚠ Invalid content type: {content_type}")
                return False

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size = os.path.getsize(save_path)
            if file_size < 1000:
                os.remove(save_path)
                print(f"        ✗✗ File too small ({file_size} bytes), possibly error page")
                return False

            print(f"        ✓ Download successful: {os.path.basename(save_path)} ({file_size} bytes)")
            return True
        except Exception as e:
            print(f"        ✗✗ Download failed: {e}")
            return False

    def get_note_images_from_api(self, note_url: str) -> List[str]:
        """Get note image URLs from API"""
        print(f"        Getting note images: {note_url}")

        # Add pre-delay to avoid being detected as bot
        time.sleep(random.uniform(2, 3.0))

        html = self._make_request(note_url)
        if not html:
            print("        Unable to get HTML content")
            return []

        # Method 1: Extract from script tags
        images = self._extract_images_from_script(html)
        if images:
            print(f"        Found {len(images)} images from script tags")
            return images

        # Method 2: Use meta tags
        images = self._extract_images_from_meta(html)
        if images:
            print(f"        Found {len(images)} images from meta tags")
            return images

        # Method 3: Simple detail parsing as fallback
        detail = self._get_note_detail_simple(html)
        if detail and detail.get("images"):
            print(f"        Found {len(detail['images'])} images from detail parsing")
            return detail["images"]

        print("        ✗✗ No images found")
        return []

    def _extract_images_from_script(self, html: str) -> List[str]:
        """Extract image URLs from script tags"""
        images = []

        patterns = [
            r'"notes":\s*(\[\[.*?\]\])',
            r'"imageList":\s*(\[.*?\])',
            r'"images":\s*(\[.*?\])',
            r'window\.__INITIAL_STATE__\s*=\s*({.*?})</script>'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, html, re.DOTALL)
            for match in matches:
                try:
                    if pattern == r'"notes":\s*(\[\[.*?\]\])':
                        from json5 import loads as json5_loads
                        notes_data = json5_loads(match)
                        if len(notes_data) > 1 and notes_data[1]:
                            for note in notes_data[1]:
                                cover_info = note.get('noteCard', {}).get('cover', {})
                                if cover_info.get('urlDefault'):
                                    images.append(cover_info['urlDefault'])
                                if cover_info.get('infoList'):
                                    for info in cover_info['infoList']:
                                        if info.get('url'):
                                            images.append(info['url'])
                    else:
                        data = json.loads(match)
                        images.extend(self._find_images_in_json(data))

                except Exception as e:
                    continue

        return images

    def _extract_images_from_meta(self, html: str) -> List[str]:
        """Extract images from meta tags"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        images = []

        for meta in soup.find_all('meta', property='og:image'):
            if meta.get('content'):
                images.append(meta['content'])

        for meta in soup.find_all('meta', attrs={'name': 'twitter:image'}):
            if meta.get('content'):
                images.append(meta['content'])

        return images

    def _get_note_detail_simple(self, html: str) -> Optional[Dict]:
        """Simple note detail extraction as fallback method"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        def get_meta(name, multi=False):
            tags = soup.find_all("meta", attrs={"name": name})
            if not tags:
                return [] if multi else None
            return [t["content"].strip() for t in tags if "content" in t.attrs] if multi else tags[0]["content"].strip()

        return {
            "title": get_meta("og:title"),
            "description": get_meta("description"),
            "images": get_meta("og:image", multi=True),
            "note_type": get_meta("og:type"),
            "video_url": get_meta("og:video"),
        }

    def _find_images_in_json(self, data: Dict) -> List[str]:
        """Recursively find image URLs in JSON"""
        images = []

        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() in ['url', 'image', 'cover', 'pic'] and isinstance(value, str):
                    if value.startswith('http') and any(ext in value.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        images.append(value)
                elif isinstance(value, (dict, list)):
                    images.extend(self._find_images_in_json(value))
        elif isinstance(data, list):
            for item in data:
                images.extend(self._find_images_in_json(item))

        return images

    async def process_redbook_content(self, content_data, folder_path, content_type, redbook_id=None):
        """Process redbook content (videos and image notes, excluding cover images)"""
        title = content_data.get('title', 'untitled')
        content = content_data.get('content', '')
        videos = content_data.get('videos', '')
        note_url = content_data.get('note_url', '')

        # Create folder with redbook_id or clean title name
        if redbook_id:
            folder_name = str(redbook_id)
        else:
            folder_name = self._sanitize_filename(title)

        item_folder = os.path.join(folder_path, folder_name)
        os.makedirs(item_folder, exist_ok=True)

        # Save text content
        text_success = self._save_text_content(title, content, item_folder)

        media_success = False

        # Check if it's a video note
        if videos and videos.strip():
            # Download video (no cover image)
            video_success = await self._download_video(videos, item_folder, title)
            media_success = video_success
        else:
            # Download images for image notes (no cover images)
            if note_url:
                image_success = await self._download_images_for_note(note_url, item_folder, title)
                media_success = image_success

        if text_success or media_success:
            print(f"[SUCCESS] {content_type} content processed: {folder_name}")
            return True
        else:
            print(f"[ERROR] {content_type} content processing failed: {folder_name}")
            return False

    def _save_text_content(self, title, content, save_path):
        """Save title and content as txt file"""
        try:
            # Clean content
            content_without_tags = re.sub(r'#\w+', '', content)
            content_without_tags = re.sub(r'\s+', ' ', content_without_tags).strip()

            if not content_without_tags:
                content_without_tags = 'No content'

            clean_title = self._sanitize_filename(title)
            filename = f"{clean_title}.txt"
            filepath = os.path.join(save_path, filename)

            # Limit content to first 2000 characters
            if len(content_without_tags) > 2000:
                content_without_tags = content_without_tags[:2000]
                print(f"[INFO] Content truncated to first 2000 characters: {filename}")

            # Combine title and content
            full_content = f"Title: {title}\n\nContent: {content_without_tags}"

            # Write content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(full_content)

            print(f"[SUCCESS] Text saved: {filename}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to save text file: {e}")
            return False

    async def _download_video(self, video_url, save_path, title):
        """Download video file"""
        try:
            # Check video limit (though usually only 1 video per note)
            if self.max_videos_per_note <= 0:
                print(f"        [INFO] Video download disabled (max_videos_per_note=0)")
                return False

            # Check for invalid video URL values (including 'nan' from pandas)
            if not video_url or video_url.strip() == '' or video_url.lower() == 'nan':
                print(f"        [INFO] No valid video URL provided (got: '{video_url}')")
                return False

            normalized_url = self._normalize_url(video_url)
            if not normalized_url:
                print(f"        ✗✗ Invalid video URL: {video_url}")
                return False

            ext = '.mp4'
            filename = f"video_0{ext}"
            filepath = os.path.join(save_path, filename)

            print(f"        Downloading video: {normalized_url[:100]}...")
            success = await self.download_file(normalized_url, filepath)
            return success

        except Exception as e:
            print(f"        ✗✗ Video download failed: {e}")
            return False

    async def _download_images_for_note(self, note_url, save_path, title):
        """Download images for image note (excluding cover images)"""
        try:
            image_urls = self.get_note_images_from_api(note_url)

            if not image_urls:
                print("        ✗✗ No images found")
                return False

            # Filter out cover images (typically the first image or images with 'cover' in URL)
            filtered_images = []
            for i, img_url in enumerate(image_urls):
                # Skip first image as it's often the cover
                if i == 0:
                    continue
                # Skip images with cover-related keywords
                if any(keyword in img_url.lower() for keyword in ['cover', 'thumb', 'preview']):
                    continue
                filtered_images.append(img_url)

            # If no images left after filtering, use original list but skip first one
            if not filtered_images and len(image_urls) > 1:
                filtered_images = image_urls[1:]
            elif not filtered_images:
                print("        ✗✗ No non-cover images found")
                return False

            # Limit the number of images to download
            if len(filtered_images) > self.max_images_per_note:
                print(f"        [INFO] Limiting images from {len(filtered_images)} to {self.max_images_per_note}")
                filtered_images = filtered_images[:self.max_images_per_note]

            download_tasks = []

            for idx, img_url in enumerate(filtered_images):
                normalized_url = self._normalize_url(img_url)
                if not normalized_url:
                    continue

                ext = self._get_file_extension(normalized_url)
                filename = f"image_{idx}{ext}"
                filepath = os.path.join(save_path, filename)

                download_tasks.append((normalized_url, filepath))

            if not download_tasks:
                return False

            # Download images concurrently
            success_count = 0
            semaphore = asyncio.Semaphore(1)  # Limit concurrent downloads

            async def limited_download(url, path):
                async with semaphore:
                    return await self.download_file(url, path)

            tasks = [limited_download(url, path) for url, path in download_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"        ✗ Download exception: {result}")
                elif result:
                    success_count += 1

            print(f"        ✓ Downloaded {success_count}/{len(download_tasks)} images")
            return success_count > 0

        except Exception as e:
            print(f"        ✗✗ Image download failed: {e}")
            return False

    async def download_validation_items(self, user_internal_id: int, user_real_id: str, valid_matrix, item_map,
                                      download_dir: str, content_mapping: dict):
        """Download and process validation items for a user"""
        validation_items = self.get_user_validation_items(user_internal_id, valid_matrix, item_map)

        if not validation_items:
            print(f"⚠️  No validation items found for user {user_real_id}")
            return []

        print(f"📥 Processing {len(validation_items)} validation items for user {user_real_id}")

        # Create validation folder
        validation_dir = os.path.join(download_dir, str(user_real_id), "validation")
        os.makedirs(validation_dir, exist_ok=True)

        # Process validation items
        successful_downloads = 0

        for redbook_id in validation_items:
            # Use redbook_id to lookup in content mapping
            matching_content = content_mapping.get(redbook_id)

            if matching_content:
                try:
                    success = await self.process_redbook_content(
                        matching_content,
                        validation_dir,
                        "validation",
                        redbook_id
                    )
                    if success:
                        successful_downloads += 1

                    # Add small delay to avoid overwhelming the system
                    await asyncio.sleep(0.2)

                except Exception as e:
                    print(f"[ERROR] Error processing validation content {redbook_id}: {e}")
            else:
                print(f"[WARNING] No content found for validation item '{redbook_id}'")

        print(f"✅ Processed {successful_downloads}/{len(validation_items)} validation items for user {user_real_id}")
        return validation_items

    async def download_test_items(self, user_internal_id: int, user_real_id: str, test_matrix, item_map,
                                  download_dir: str, content_mapping: dict):
        """Download and process test items for a user"""
        test_items = self.get_user_test_items(user_internal_id, test_matrix, item_map)

        if not test_items:
            print(f"⚠️  No test items found for user {user_real_id}")
            return []

        print(f"📥 Processing {len(test_items)} test items for user {user_real_id}")

        # Create test folder
        test_dir = os.path.join(download_dir, str(user_real_id), "test")
        os.makedirs(test_dir, exist_ok=True)

        # Process test items
        successful_downloads = 0

        for redbook_id in test_items:
            # Use redbook_id to lookup in content mapping
            matching_content = content_mapping.get(redbook_id)

            if matching_content:
                try:
                    success = await self.process_redbook_content(
                        matching_content,
                        test_dir,
                        "test",
                        redbook_id
                    )
                    if success:
                        successful_downloads += 1

                    # Add small delay to avoid overwhelming the system
                    await asyncio.sleep(0.2)

                except Exception as e:
                    print(f"[ERROR] Error processing test content {redbook_id}: {e}")
            else:
                print(f"[WARNING] No content found for test item '{redbook_id}'")

        print(f"✅ Processed {successful_downloads}/{len(test_items)} test items for user {user_real_id}")
        return test_items

    async def process_content_batch(self, content_mapping, item_ids, folder_path, content_type):
        """Process a batch of content items concurrently

        Args:
            content_mapping: Dictionary mapping item_id to content data
            item_ids: List of redbook IDs to process
            folder_path: Base folder path for this content type
            content_type: "推荐" or "历史"

        Returns:
            Number of successfully processed items
        """
        print(f"[INFO] Starting concurrent processing of {content_type} content ({len(item_ids)} items)...")

        # Create tasks for concurrent processing within this batch
        batch_tasks = []
        for i, redbook_id in enumerate(item_ids):
            # Use redbook_id to lookup in content mapping
            matching_content = content_mapping.get(redbook_id)

            if matching_content:
                task = self.process_single_content(
                    matching_content, folder_path, content_type, i+1, len(item_ids), redbook_id
                )
                batch_tasks.append(task)
            else:
                print(f"[WARNING] No {content_type} content found for '{redbook_id}'")

        if not batch_tasks:
            print(f"[WARNING] No valid tasks for {content_type} content batch")
            return 0

        # Process batch concurrently with limited concurrency to avoid overwhelming the system
        semaphore = asyncio.Semaphore(1)  # Limit to 3 concurrent downloads per batch

        async def limited_process(task):
            async with semaphore:
                return await task

        limited_tasks = [limited_process(task) for task in batch_tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)

        # Count successful downloads
        successful_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[ERROR] {content_type} content {i+1} processing failed: {result}")
            elif result:
                successful_count += 1

        print(f"[SUCCESS] {content_type} content batch completed: {successful_count}/{len(batch_tasks)} successful")
        return successful_count

    async def process_single_content(self, content_data, folder_path, content_type, index, total, redbook_id=None):
        """Process a single content item

        Args:
            content_data: Content data dictionary
            folder_path: Folder path for this content type
            content_type: "推荐" or "历史"
            index: Current item index
            total: Total items in batch
            redbook_id: Redbook ID to use as folder name

        Returns:
            True if successful, False otherwise
        """
        title = content_data['title'][:30] + "..." if len(content_data['title']) > 30 else content_data['title']
        display_name = redbook_id if redbook_id else title
        print(f"[INFO] Processing {content_type} content {index}/{total}: {display_name}")

        try:
            success = await self.process_redbook_content(
                content_data,
                folder_path,
                content_type,
                redbook_id
            )

            # Add small delay to avoid overwhelming the server
            await asyncio.sleep(0.3)

            return success

        except Exception as e:
            print(f"[ERROR] Error processing {content_type} content: {e}")
            return False

    async def redbook_test(self):
        """
        Test redbook content download with concurrent processing
        """
        print("\n" + "="*50)
        print("[INFO] 开始 Redbook 内容下载测试 (并发模式)...")
        print("="*50)

        origin_work_dir = os.path.dirname(os.getcwd())
        download_dir = os.path.join(origin_work_dir, "download", "redbook")
        os.makedirs(download_dir, exist_ok=True)

        print("[INFO] 用户数量：", configs['data']['tst_user_num'])
        random_ids = torch.randint(0, configs['data']['tst_user_num'], (30,))
        print(f"[INFO] 随机用户ID: {random_ids}")

        print("[INFO] 生成推荐预测中...", flush=True)
        sample_preds = self.model.sample_predict(random_ids)
        print(f"[SUCCESS] 推荐预测完成: {sample_preds.shape}", flush=True)

        # 自动开始下载
        print(f"\n[INFO] 开始自动下载内容...", flush=True)

        print("[INFO] 开始加载数据文件...")
        with open("datasets/general_cf/redbook/item_map.json", "r", encoding="utf-8") as f:
            item_map = json.load(f)
        with open("datasets/general_cf/redbook/user_map.json", "r", encoding="utf-8") as f:
            user_map = json.load(f)

        # Load training matrix to get user historical interactions
        import pickle
        with open("datasets/general_cf/redbook/train_matrix.pkl", "rb") as f:
            train_mat = pickle.load(f)

        # Load validation matrix for validation items download
        try:
            with open("datasets/general_cf/redbook/valid_matrix.pkl", "rb") as f:
                valid_mat = pickle.load(f)
            print(f"[INFO] 加载验证矩阵成功: {valid_mat.shape}")
        except Exception as e:
            print(f"[WARNING] 无法加载验证矩阵: {e}")
            valid_mat = None

        # Load test matrix for test items download
        try:
            with open("datasets/general_cf/redbook/test_matrix.pkl", "rb") as f:
                test_mat = pickle.load(f)
            print(f"[INFO] 加载测试矩阵成功: {test_mat.shape}")
        except Exception as e:
            print(f"[WARNING] 无法加载测试矩阵: {e}")
            test_mat = None

        for i, user_id in enumerate(random_ids):
            user_id_int = int(user_id.item())
            real_user_id = user_map[str(user_id_int)]
            print("[INFO] 用户ID: ", user_id_int, "[INFO] 真实用户ID: ", real_user_id)

            user_download_dir = os.path.join(download_dir, str(real_user_id))
            os.makedirs(user_download_dir, exist_ok=True)

            # Get recommended items (item_map contains douban IDs)
            top_item_indices = sample_preds[i].tolist()
            recommended_redbook_ids = [item_map[str(item_id)] for item_id in top_item_indices if str(item_id) in item_map]
            print(f"[INFO] 用户 {user_id_int} 推荐内容: {recommended_redbook_ids}", flush=True)

            # Get user's historical interactions from training matrix
            user_interactions = train_mat[user_id_int].nonzero()[1]  # Get column indices (item indices)
            historical_redbook_ids = [item_map[str(item_id)] for item_id in user_interactions if str(item_id) in item_map]
            print(f"[INFO] 用户 {user_id_int} 历史内容 ({len(historical_redbook_ids)} 个): {historical_redbook_ids[:10]}...", flush=True)  # Show first 10

            # Combine recommended and historical items for download
            all_redbook_ids = recommended_redbook_ids + historical_redbook_ids
            print(f"[INFO] 用户 {user_id_int} 总计下载: {len(all_redbook_ids)} 个内容 (推荐: {len(recommended_redbook_ids)}, 历史: {len(historical_redbook_ids)})", flush=True)

            # Load douban data to get actual content
            redbook_data_dir = os.path.join(origin_work_dir, "dataset", "redbook", str(real_user_id))

            if not os.path.exists(redbook_data_dir):
                print(f"[WARNING] 用户 {real_user_id} 的数据目录不存在: {redbook_data_dir}")
                continue

            # Create separate folders for recommended and historical items
            recommended_folder = os.path.join(user_download_dir, "recommended")
            historical_folder = os.path.join(user_download_dir, "historical")
            os.makedirs(recommended_folder, exist_ok=True)
            os.makedirs(historical_folder, exist_ok=True)

            # Process CSV files in user directory to build content mapping for historical items
            csv_files = [f for f in os.listdir(redbook_data_dir) if f.endswith('.csv')]

            # Build a mapping from douban ID to content data
            content_mapping = {}

            # First, load content from user's CSV files (historical items)
            for csv_file in csv_files:
                csv_path = os.path.join(redbook_data_dir, csv_file)

                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)

                    print(f"[INFO] 加载用户CSV文件: {csv_file}, 包含 {len(df)} 条记录")

                    for idx, row in df.iterrows():
                        title = str(row.get('title', f'untitled_{idx}'))
                        redbook_id = str(row.get('redbookID', ''))

                        if redbook_id:  # Only add if redbook_id exists
                            # Handle NaN values properly
                            videos_val = row.get('videos', '')
                            if pd.isna(videos_val):
                                videos_val = ''

                            note_url_val = row.get('笔记URL', '')
                            if pd.isna(note_url_val):
                                note_url_val = ''

                            content_val = row.get('content', '')
                            if pd.isna(content_val):
                                content_val = ''

                            content_data = {
                                'title': title,
                                'content': str(content_val),
                                'videos': str(videos_val),
                                'note_url': str(note_url_val)
                            }

                            # Use redbook ID as the key for mapping
                            content_mapping[redbook_id] = content_data

                except Exception as e:
                    print(f"[ERROR] 加载用户CSV文件 {csv_file} 时出错: {e}")
                    continue

            # Second, load content from enhanced redbook_mapping.json for recommended items
            print(f"[INFO] 从redbook_mapping.json 加载推荐内容...")

            # Find recommended items that are not in user's CSV data
            missing_recommended_ids = [redbook_id for redbook_id in recommended_redbook_ids
                                     if redbook_id not in content_mapping]

            if missing_recommended_ids:
                print(f"[INFO] 需要从 redbook_mapping.json 加载 {len(missing_recommended_ids)} 个推荐内容")

                # Load redbook_mapping.json
                redbook_mapping_file = os.path.join(origin_work_dir, "redbook_mapping.json")
                if os.path.exists(redbook_mapping_file):
                    try:
                        with open(redbook_mapping_file, 'r', encoding='utf-8') as f:
                            redbook_mapping = json.load(f)

                        loaded_count = 0
                        for redbook_id in missing_recommended_ids:
                            # For redbook, the key is the redbook_id without .json extension
                            if redbook_id in redbook_mapping:
                                item_data = redbook_mapping[redbook_id]
                                if isinstance(item_data, dict):
                                    # New enhanced format
                                    content_data = {
                                        'title': item_data.get('title', f'untitled_{redbook_id}'),
                                        'content': item_data.get('content', ''),
                                        'videos': item_data.get('videos', ''),
                                        'note_url': item_data.get('note_url', '')
                                    }
                                else:
                                    # Old format (just title)
                                    content_data = {
                                        'title': item_data,
                                        'content': '',
                                        'videos': '',
                                        'note_url': ''
                                    }

                                content_mapping[redbook_id] = content_data
                                loaded_count += 1
                                print(f"[INFO] 从 mapping 加载推荐内容: {redbook_id} - {content_data['title'][:50]}...")
                            else:
                                print(f"[WARNING] 推荐内容 {redbook_id} 在 mapping 中未找到")

                        print(f"[SUCCESS] 从 redbook_mapping.json 成功加载 {loaded_count}/{len(missing_recommended_ids)} 个推荐内容")

                    except Exception as e:
                        print(f"[ERROR] 加载 redbook_mapping.json 时出错: {e}")
                else:
                    print(f"[WARNING] redbook_mapping.json 文件不存在: {redbook_mapping_file}")
            else:
                print(f"[INFO] 所有推荐内容都已在用户CSV数据中找到，无需额外加载")


            # Create concurrent tasks for recommended and historical content
            tasks = []
            start_time = time.time()

            # Create task for recommended content processing
            if recommended_redbook_ids:
                recommended_task = self.process_content_batch(
                    content_mapping, recommended_redbook_ids, recommended_folder, "推荐"
                )
                tasks.append(recommended_task)

            # Create task for historical content processing
            if historical_redbook_ids:
                historical_task = self.process_content_batch(
                    content_mapping, historical_redbook_ids, historical_folder, "历史"
                )
                tasks.append(historical_task)

            print(f"\n[INFO] 开始并发处理 {len(tasks)} 个任务...")

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            total_time = end_time - start_time

            # Process results
            total_successful = 0
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"[ERROR] 任务 {j+1} 执行失败: {result}")
                else:
                    total_successful += result

            print(f"\n[SUMMARY] 用户 {user_id_int} 下载总结:")
            print(f"  [SUCCESS] 成功处理: {total_successful} 个内容")
            print(f"  [INFO] 推荐内容: {len(recommended_redbook_ids)} 个")
            print(f"  [INFO] 历史内容: {len(historical_redbook_ids)} 个")
            print(f"  [INFO] 总耗时: {total_time:.2f} 秒")
            print(f"  [INFO] CSV文件数: {len(csv_files)} 个")

            # Download validation items for this user
            if valid_mat is not None:
                print(f"  [INFO] 📥 开始处理用户 {user_id_int} 的 validation 数据...")
                await self.download_validation_items(
                    user_id_int, real_user_id, valid_mat, item_map, download_dir, content_mapping
                )
            else:
                print(f"  [WARNING] ⚠️  跳过 validation 数据处理 (未找到验证矩阵)")

            # Download test items for this user
            if test_mat is not None:
                print(f"  [INFO] 📥 开始处理用户 {user_id_int} 的 test 数据...")
                await self.download_test_items(
                    user_id_int, real_user_id, test_mat, item_map, download_dir, content_mapping
                )
            else:
                print(f"  [WARNING] ⚠️  跳过 test 数据处理 (未找到测试矩阵)")

            print("="*50, flush=True)

        # Print request statistics
        self.print_request_statistics()

    def print_request_statistics(self):
        """Print request statistics"""
        print(f"\n[STATISTICS] Request Summary:")
        print(f"  [SUCCESS] Successful requests: {self.successful_requests}")
        print(f"  [FAILED] Failed requests: {self.failed_requests}")
        total_requests = self.successful_requests + self.failed_requests
        if total_requests > 0:
            success_rate = (self.successful_requests / total_requests) * 100
            print(f"  [RATE] Success rate: {success_rate:.1f}%")
        print("="*50, flush=True)


class HupuDownloader:
    """
    Hupu content downloader for images

    Usage example:
        downloader = HupuDownloader(model)
        await downloader.hupu_test()
    """
    def __init__(self, model):
        self.model = model
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session.headers.update(self.headers)
        self.timeout = 30
        self.max_retries = 3

        # Download limits
        self.max_images_per_post = 3  # Maximum images per post

        # Request statistics
        self.successful_requests = 0
        self.failed_requests = 0

    def set_download_limits(self, max_images_per_post=3):
        """Set download limits for images

        Args:
            max_images_per_post: Maximum number of images to download per post (default: 3)
        """
        self.max_images_per_post = max_images_per_post
        print(f"✅ Set download limits: max_images={max_images_per_post}")

    def get_user_validation_items(self, user_internal_id: int, valid_matrix, item_map) -> list:
        """Get validation items for a user (convert from internal ID to real item IDs)"""
        if valid_matrix is None or user_internal_id >= valid_matrix.shape[0]:
            return []

        # Get item indices that this user interacted with in validation set
        valid_item_indices = valid_matrix[user_internal_id].nonzero()[1]

        # Convert to real item IDs
        valid_items = []
        for item_idx in valid_item_indices:
            real_item_id = item_map.get(str(item_idx))
            if real_item_id:
                valid_items.append(real_item_id)

        return valid_items

    def get_user_test_items(self, user_internal_id: int, test_matrix, item_map) -> list:
        """Get test items for a user (convert from internal ID to real item IDs)"""
        if test_matrix is None or user_internal_id >= test_matrix.shape[0]:
            return []

        # Get item indices that this user interacted with in test set
        test_item_indices = test_matrix[user_internal_id].nonzero()[1]

        # Convert to real item IDs
        test_items = []
        for item_idx in test_item_indices:
            real_item_id = item_map.get(str(item_idx))
            if real_item_id:
                test_items.append(real_item_id)

        return test_items

    def _sanitize_filename(self, filename: str) -> str:
        """Clean filename of invalid characters"""
        if not filename or filename.strip() == '':
            return "无标题"

        filename = re.sub(r'[\uFEFF\u200B-\u200D\uFFFC]', '', filename)
        invalid_chars = r'<>:"/\\|?*#\r\n\t'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Limit filename length
        if len(filename) > 100:
            filename = filename[:100] + "..."

        return filename.strip()

    def get_file_extension(self, url):
        """Get file extension from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path
        if '.' in path:
            ext = path.split('.')[-1].split('?')[0]  # Remove query parameters
            return f".{ext}" if ext else ".jpg"
        return ".jpg"

    def generate_filename(self, hupu_id, url, index=0):
        """Generate filename for image"""
        ext = self.get_file_extension(url)
        if index == 0:
            return f"{hupu_id}{ext}"
        else:
            return f"{hupu_id}_{index}{ext}"

    async def download_hupu_images(self, images_str, save_path, hupu_id, max_retries=3, max_images=None):
        """Download images from semicolon-separated URLs with concurrent processing

        Args:
            images_str: Semicolon-separated image URLs
            save_path: Path to save the images (directly in this folder)
            hupu_id: Hupu post ID for filename generation
            max_retries: Maximum retry attempts
            max_images: Maximum number of images to download per item
        """
        if not images_str or images_str.strip() == '' or str(images_str).lower() == 'nan':
            return True

        # Split images by semicolon and filter out empty strings and invalid URLs
        image_urls = []
        for url in images_str.split(';'):
            url = url.strip()
            if url and url.lower() != 'nan' and (url.startswith('http://') or url.startswith('https://')):
                image_urls.append(url)

        if not image_urls:
            return True

        # Limit the number of images to download
        max_imgs = max_images or self.max_images_per_post
        if len(image_urls) > max_imgs:
            print(f"[INFO] 图片数量限制: 从 {len(image_urls)} 张中选择前 {max_imgs} 张下载")
            image_urls = image_urls[:max_imgs]

        print(f"[INFO] 并发下载 {len(image_urls)} 张图片到 {save_path}")

        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Create concurrent download tasks
        download_tasks = []
        for i, image_url in enumerate(image_urls):
            task = self.download_single_image(image_url, save_path, hupu_id, i, max_retries)
            download_tasks.append(task)

        # Limit concurrent downloads to avoid overwhelming the server
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent image downloads

        async def limited_download(task):
            async with semaphore:
                return await task

        limited_tasks = [limited_download(task) for task in download_tasks]

        # Execute all downloads concurrently
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)

        # Count successful downloads
        successful_downloads = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[ERROR] 图片 {i+1} 下载异常: {result}")
            elif result:
                successful_downloads += 1

        print(f"[SUCCESS] 图片下载完成: {successful_downloads}/{len(image_urls)} 张成功")
        return successful_downloads > 0

    async def download_single_image(self, image_url, save_path, hupu_id, image_index, max_retries=3):
        """Download a single image with retry mechanism

        Args:
            image_url: URL of the image to download
            save_path: Directory to save the image
            hupu_id: Hupu post ID for filename generation
            image_index: Index of the image (for filename)
            max_retries: Maximum retry attempts

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid being detected as bot
                if attempt > 0:
                    delay = random.uniform(1, 3)  # 1-3 seconds delay
                    print(f"[INFO] 重试下载图片 {image_index+1}, 等待 {delay:.1f} 秒...")
                    await asyncio.sleep(delay)

                # Generate filename
                filename = self.generate_filename(hupu_id, image_url, image_index)
                filepath = os.path.join(save_path, filename)

                # Check if file already exists
                if os.path.exists(filepath):
                    print(f"[INFO] 图片已存在，跳过: {filename}")
                    return True

                # Download image using requests
                response = self.session.get(image_url, timeout=self.timeout, stream=True)
                response.raise_for_status()

                # Check if it's an image
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    print(f"[WARNING] 不是图片类型: {content_type}")
                    return False

                # Save image
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                print(f"[SUCCESS] 下载成功: {filename}")
                self.successful_requests += 1
                return True

            except requests.exceptions.RequestException as e:
                self.failed_requests += 1
                if attempt == max_retries - 1:
                    print(f"[ERROR] 图片下载失败 (最终尝试): {str(e)}")
                    return False
                else:
                    print(f"[WARNING] 图片下载失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

            except Exception as e:
                self.failed_requests += 1
                print(f"[ERROR] 未知错误: {str(e)}")
                return False

        return False

    async def download_validation_items(self, user_internal_id: int, user_real_id: str, valid_matrix, item_map,
                                      download_dir: str, content_mapping: dict):
        """Download and process validation items for a user"""
        validation_items = self.get_user_validation_items(user_internal_id, valid_matrix, item_map)

        if not validation_items:
            print(f"⚠️  No validation items found for user {user_real_id}")
            return []

        print(f"📥 Processing {len(validation_items)} validation items for user {user_real_id}")

        # Create validation folder
        validation_dir = os.path.join(download_dir, str(user_real_id), "validation")
        os.makedirs(validation_dir, exist_ok=True)

        # Process validation items
        successful_downloads = 0

        for hupu_id in validation_items:
            # Use hupu_id to lookup in content mapping
            matching_content = content_mapping.get(hupu_id)

            if matching_content:
                try:
                    success = await self.process_hupu_content(
                        matching_content,
                        validation_dir,
                        "validation",
                        hupu_id
                    )
                    if success:
                        successful_downloads += 1

                    # Add small delay to avoid overwhelming the system
                    await asyncio.sleep(0.2)

                except Exception as e:
                    print(f"[ERROR] 处理validation内容时出错 {hupu_id}: {e}")
            else:
                print(f"[WARNING] 未找到validation内容 '{hupu_id}' 的对应数据")

        print(f"✅ Processed {successful_downloads}/{len(validation_items)} validation items for user {user_real_id}")
        return validation_items

    async def download_test_items(self, user_internal_id: int, user_real_id: str, test_matrix, item_map,
                                  download_dir: str, content_mapping: dict):
        """Download and process test items for a user"""
        test_items = self.get_user_test_items(user_internal_id, test_matrix, item_map)

        if not test_items:
            print(f"⚠️  No test items found for user {user_real_id}")
            return []

        print(f"📥 Processing {len(test_items)} test items for user {user_real_id}")

        # Create test folder
        test_dir = os.path.join(download_dir, str(user_real_id), "test")
        os.makedirs(test_dir, exist_ok=True)

        # Process test items
        successful_downloads = 0

        for hupu_id in test_items:
            # Use hupu_id to lookup in content mapping
            matching_content = content_mapping.get(hupu_id)

            if matching_content:
                try:
                    success = await self.process_hupu_content(
                        matching_content,
                        test_dir,
                        "test",
                        hupu_id
                    )
                    if success:
                        successful_downloads += 1

                    # Add small delay to avoid overwhelming the system
                    await asyncio.sleep(0.2)

                except Exception as e:
                    print(f"[ERROR] 处理test内容时出错 {hupu_id}: {e}")
            else:
                print(f"[WARNING] 未找到test内容 '{hupu_id}' 的对应数据")

        print(f"✅ Processed {successful_downloads}/{len(test_items)} test items for user {user_real_id}")
        return test_items

    def _save_text_content(self, title, content, save_path):
        """Save title and content as txt file (similar to redbook/douban implementation)"""
        try:
            # Clean content - remove hashtags and normalize whitespace
            import re
            content_without_tags = re.sub(r'#\w+', '', str(content))
            content_without_tags = re.sub(r'\s+', ' ', content_without_tags).strip()

            if not content_without_tags or content_without_tags.lower() == 'nan':
                content_without_tags = 'No content'

            clean_title = self._sanitize_filename(title)
            filename = f"{clean_title}.txt"
            filepath = os.path.join(save_path, filename)

            # Limit content to first 2000 characters to avoid extremely long files
            if len(content_without_tags) > 2000:
                content_without_tags = content_without_tags[:2000] + "..."
                print(f"[INFO] Content truncated to first 2000 characters: {filename}")

            # Combine title and content (similar to redbook/douban format)
            full_content = f"Title: {title}\n\nContent: {content_without_tags}"

            # Write content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(full_content)

            print(f"[SUCCESS] Text saved: {filename}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to save text file: {e}")
            return False

    async def process_hupu_content(self, content_data, folder_path, content_type, hupu_id=None):
        """Process hupu content (images and text)

        Args:
            content_data: Dictionary containing content information
            folder_path: Base folder path to save content
            content_type: Type of content (推荐/历史/validation/test)
            hupu_id: Hupu ID to use as folder name, if None use title

        Returns:
            True if successful, False otherwise
        """
        try:
            title = content_data.get('title', '无标题')
            content = content_data.get('content', '')
            images = content_data.get('images', '')

            # Create folder with hupu_id or clean title name
            if hupu_id:
                folder_name = str(hupu_id)
            else:
                folder_name = self._sanitize_filename(title)

            item_folder = os.path.join(folder_path, folder_name)
            os.makedirs(item_folder, exist_ok=True)

            # Save text content (title + content)
            text_success = self._save_text_content(title, content, item_folder)

            # Download images if available
            image_success = True
            if images:
                image_success = await self.download_hupu_images(images, item_folder, hupu_id or folder_name, max_images=self.max_images_per_post)

            return text_success and image_success

        except Exception as e:
            print(f"[ERROR] 处理内容失败: {e}")
            return False

    async def process_content_batch(self, content_mapping: dict, item_ids: list, folder_path: str, content_type: str):
        """Process a batch of content items

        Args:
            content_mapping: Dictionary mapping item IDs to content data
            item_ids: List of hupu IDs to process
            folder_path: Base folder path
            content_type: Type of content (推荐/历史)

        Returns:
            Number of successfully processed items
        """
        if not item_ids:
            print(f"[WARNING] 没有{content_type}内容需要处理")
            return 0

        print(f"[INFO] 开始处理 {len(item_ids)} 个{content_type}内容...")

        successful_count = 0
        for i, hupu_id in enumerate(item_ids):
            # Use hupu_id to lookup in content mapping
            matching_content = content_mapping.get(hupu_id)

            if matching_content:
                try:
                    success = await self.process_single_content(
                        matching_content, folder_path, content_type, i+1, len(item_ids), hupu_id
                    )
                    if success:
                        successful_count += 1
                except Exception as e:
                    print(f"[ERROR] 处理{content_type}内容时出错: {e}")
            else:
                print(f"[WARNING] 未找到{content_type}内容 '{hupu_id}' 的对应数据")

        return successful_count

    async def process_single_content(self, content_data, folder_path, content_type, index, total, hupu_id=None):
        """Process a single content item with progress tracking

        Args:
            content_data: Content data dictionary
            folder_path: Base folder path
            content_type: Type of content (推荐/历史)
            index: Current index
            total: Total number of items
            hupu_id: Hupu ID to use as folder name
        """
        try:
            title = content_data.get('title', '无标题')
            display_name = hupu_id if hupu_id else title

            print(f"[INFO] 处理{content_type}内容 ({index}/{total}): {display_name[:50]}...")

            success = await self.process_hupu_content(
                content_data,
                folder_path,
                content_type,
                hupu_id
            )

            if success:
                print(f"[SUCCESS] ✅ {content_type}内容处理成功: {display_name[:30]}...")
            else:
                print(f"[WARNING] ⚠️  {content_type}内容处理部分失败: {display_name[:30]}...")

            return success

        except Exception as e:
            print(f"[ERROR] 处理{content_type}内容时出错: {e}")
            return False

    async def hupu_test(self):
        """
        Test hupu content download with concurrent processing
        """
        print("\n" + "="*50)
        print("[INFO] 开始 Hupu 内容下载测试 (并发模式)...")
        print("="*50)

        origin_work_dir = os.path.dirname(os.getcwd())
        download_dir = os.path.join(origin_work_dir, "download", "hupu")
        os.makedirs(download_dir, exist_ok=True)

        print("[INFO] 用户数量：", configs['data']['tst_user_num'])
        random_ids = torch.randint(0, configs['data']['tst_user_num'], (10,))
        print(f"[INFO] 随机用户ID: {random_ids}")

        print("[INFO] 生成推荐预测中...", flush=True)
        sample_preds = self.model.sample_predict(random_ids)
        print(f"[SUCCESS] 推荐预测完成: {sample_preds.shape}", flush=True)

        # 自动开始下载
        print(f"\n[INFO] 开始自动下载内容...", flush=True)

        print("[INFO] 开始加载数据文件...")
        with open("datasets/general_cf/hupu/item_map.json", "r", encoding="utf-8") as f:
            item_map = json.load(f)
        with open("datasets/general_cf/hupu/user_map.json", "r", encoding="utf-8") as f:
            user_map = json.load(f)

        # Load training matrix to get user historical interactions
        import pickle
        with open("datasets/general_cf/hupu/train_matrix.pkl", "rb") as f:
            train_mat = pickle.load(f)

        # Load validation matrix for validation items download
        try:
            with open("datasets/general_cf/hupu/valid_matrix.pkl", "rb") as f:
                valid_mat = pickle.load(f)
            print(f"[INFO] 加载验证矩阵成功: {valid_mat.shape}")
        except Exception as e:
            print(f"[WARNING] 无法加载验证矩阵: {e}")
            valid_mat = None

        # Load test matrix for test items download
        try:
            with open("datasets/general_cf/hupu/test_matrix.pkl", "rb") as f:
                test_mat = pickle.load(f)
            print(f"[INFO] 加载测试矩阵成功: {test_mat.shape}")
        except Exception as e:
            print(f"[WARNING] 无法加载测试矩阵: {e}")
            test_mat = None

        for i, user_id in enumerate(random_ids):
            user_id_int = int(user_id.item())
            real_user_id = user_map[str(user_id_int)]
            print("[INFO] 用户ID: ", user_id_int, "[INFO] 真实用户ID: ", real_user_id)

            user_download_dir = os.path.join(download_dir, str(real_user_id))
            os.makedirs(user_download_dir, exist_ok=True)

            # Get recommended items (item_map contains hupu IDs)
            top_item_indices = sample_preds[i].tolist()
            recommended_hupu_ids = [item_map[str(item_id)] for item_id in top_item_indices if str(item_id) in item_map]
            print(f"[INFO] 用户 {user_id_int} 推荐内容: {recommended_hupu_ids}", flush=True)

            # Get user's historical interactions from training matrix
            user_interactions = train_mat[user_id_int].nonzero()[1]  # Get column indices (item indices)
            historical_hupu_ids = [item_map[str(item_id)] for item_id in user_interactions if str(item_id) in item_map]
            print(f"[INFO] 用户 {user_id_int} 历史内容 ({len(historical_hupu_ids)} 个): {historical_hupu_ids[:10]}...", flush=True)  # Show first 10

            # Combine recommended and historical items for download
            all_hupu_ids = recommended_hupu_ids + historical_hupu_ids
            print(f"[INFO] 用户 {user_id_int} 总计下载: {len(all_hupu_ids)} 个内容 (推荐: {len(recommended_hupu_ids)}, 历史: {len(historical_hupu_ids)})", flush=True)

            # Load hupu data to get actual content
            hupu_data_dir = os.path.join(origin_work_dir, "dataset", "hupu", str(real_user_id))

            if not os.path.exists(hupu_data_dir):
                print(f"[WARNING] 用户 {real_user_id} 的数据目录不存在: {hupu_data_dir}")
                continue

            # Create separate folders for recommended and historical items
            recommended_folder = os.path.join(user_download_dir, "recommended")
            historical_folder = os.path.join(user_download_dir, "historical")
            os.makedirs(recommended_folder, exist_ok=True)
            os.makedirs(historical_folder, exist_ok=True)

            # Process CSV files in user directory to build content mapping for historical items
            csv_files = [f for f in os.listdir(hupu_data_dir) if f.endswith('.csv')]

            # Build a mapping from hupu ID to content data
            content_mapping = {}

            # First, load content from user's CSV files (historical items)
            for csv_file in csv_files:
                csv_path = os.path.join(hupu_data_dir, csv_file)

                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)

                    print(f"[INFO] 加载用户CSV文件: {csv_file}, 包含 {len(df)} 条记录")

                    for idx, row in df.iterrows():
                        title = str(row.get('title', f'untitled_{idx}'))
                        hupu_id = str(row.get('hupuID', ''))

                        if hupu_id:  # Only add if hupu_id exists
                            # Note: Hupu CSV uses 'text' column for post content, not 'content'
                            content_data = {
                                'title': title,
                                'content': str(row.get('text', '')),  # 'text' is the correct column name in hupu CSV
                                'images': str(row.get('images', ''))
                            }

                            # Use hupu ID as the key for mapping
                            content_mapping[hupu_id] = content_data

                except Exception as e:
                    print(f"[ERROR] 加载用户CSV文件 {csv_file} 时出错: {e}")
                    continue

            # Second, load content from enhanced hupu_mapping.json for recommended items and items with empty content
            print(f"[INFO] 从hupu_mapping.json 加载/补充内容...")

            # Find recommended items that are not in user's CSV data
            missing_recommended_ids = [hupu_id for hupu_id in recommended_hupu_ids
                                     if hupu_id not in content_mapping]

            # Also find items with empty content that need to be supplemented from hupu_mapping.json
            empty_content_ids = [hupu_id for hupu_id, data in content_mapping.items()
                               if not data.get('content') or data.get('content') == 'nan' or data.get('content').strip() == '']

            # Combine all IDs that need data from hupu_mapping.json
            all_ids_to_load = list(set(missing_recommended_ids + empty_content_ids + historical_hupu_ids))

            if all_ids_to_load:
                print(f"[INFO] 需要从 hupu_mapping.json 加载/补充 {len(all_ids_to_load)} 个内容 (缺失推荐: {len(missing_recommended_ids)}, 空内容: {len(empty_content_ids)})")

                # Load hupu_mapping.json
                hupu_mapping_file = os.path.join(origin_work_dir, "hupu_mapping.json")
                if os.path.exists(hupu_mapping_file):
                    try:
                        with open(hupu_mapping_file, 'r', encoding='utf-8') as f:
                            hupu_mapping = json.load(f)

                        loaded_count = 0
                        supplemented_count = 0
                        for hupu_id in all_ids_to_load:
                            # For hupu, the key is the hupu_id without .json extension
                            if hupu_id in hupu_mapping:
                                item_data = hupu_mapping[hupu_id]
                                if isinstance(item_data, dict):
                                    # New enhanced format
                                    new_content_data = {
                                        'title': item_data.get('title', f'untitled_{hupu_id}'),
                                        'content': item_data.get('content', ''),
                                        'images': item_data.get('images', ''),
                                        'videos': item_data.get('videos', '')
                                    }
                                else:
                                    # Old format (just title)
                                    new_content_data = {
                                        'title': item_data,
                                        'content': '',
                                        'images': '',
                                        'videos': ''
                                    }

                                # Check if this is a new entry or supplementing existing one
                                if hupu_id not in content_mapping:
                                    content_mapping[hupu_id] = new_content_data
                                    loaded_count += 1
                                elif not content_mapping[hupu_id].get('content') or content_mapping[hupu_id].get('content') == 'nan':
                                    # Supplement empty content with data from mapping
                                    if new_content_data.get('content'):
                                        content_mapping[hupu_id]['content'] = new_content_data['content']
                                        supplemented_count += 1
                                        print(f"[INFO] 补充内容: {hupu_id} - {new_content_data['title'][:30]}...")

                        print(f"[SUCCESS] 从 hupu_mapping.json 成功加载 {loaded_count} 个新内容, 补充 {supplemented_count} 个空内容")

                    except Exception as e:
                        print(f"[ERROR] 加载 hupu_mapping.json 时出错: {e}")
                else:
                    print(f"[WARNING] hupu_mapping.json 文件不存在: {hupu_mapping_file}")
            else:
                print(f"[INFO] 所有内容都已完整，无需额外加载")


            # Create tasks for concurrent processing
            tasks = []

            # Process recommended content
            if recommended_hupu_ids:
                tasks.append(self.process_content_batch(
                    content_mapping, recommended_hupu_ids, recommended_folder, "推荐"
                ))

            # Process historical content
            if historical_hupu_ids:
                tasks.append(self.process_content_batch(
                    content_mapping, historical_hupu_ids, historical_folder, "历史"
                ))

            # Execute all tasks concurrently
            if tasks:
                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                total_time = time.time() - start_time

                # Process results
                total_successful = 0
                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"[ERROR] 任务 {j+1} 执行失败: {result}")
                    else:
                        total_successful += result

                print(f"\n[SUMMARY] 用户 {user_id_int} 下载总结:")
                print(f"  [SUCCESS] 成功处理: {total_successful} 个内容")
                print(f"  [INFO] 推荐内容: {len(recommended_hupu_ids)} 个")
                print(f"  [INFO] 历史内容: {len(historical_hupu_ids)} 个")
                print(f"  [INFO] 总耗时: {total_time:.2f} 秒")
                print(f"  [INFO] CSV文件数: {len(csv_files)} 个")

                # Download validation items for this user
                if valid_mat is not None:
                    print(f"  [INFO] 📥 开始处理用户 {user_id_int} 的 validation 数据...")
                    await self.download_validation_items(
                        user_id_int, real_user_id, valid_mat, item_map, download_dir, content_mapping
                    )
                else:
                    print(f"  [WARNING] ⚠️  跳过 validation 数据处理 (未找到验证矩阵)")

                # Download test items for this user
                if test_mat is not None:
                    print(f"  [INFO] 📥 开始处理用户 {user_id_int} 的 test 数据...")
                    await self.download_test_items(
                        user_id_int, real_user_id, test_mat, item_map, download_dir, content_mapping
                    )
                else:
                    print(f"  [WARNING] ⚠️  跳过 test 数据处理 (未找到测试矩阵)")

                print("="*50, flush=True)

        # Print request statistics
        self.print_request_statistics()

    def print_request_statistics(self):
        """Print request statistics"""
        print(f"\n[STATISTICS] Request Summary:")
        print(f"  [SUCCESS] Successful requests: {self.successful_requests}")
        print(f"  [FAILED] Failed requests: {self.failed_requests}")
        total_requests = self.successful_requests + self.failed_requests
        if total_requests > 0:
            success_rate = (self.successful_requests / total_requests) * 100
            print(f"  [RATE] Success rate: {success_rate:.1f}%")
        print("="*50, flush=True)


