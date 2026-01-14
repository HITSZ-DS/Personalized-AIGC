
import json
import os
import pandas as pd
import time
import csv
import asyncio
import random
import subprocess
import shutil
from typing import List, Dict, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bilibli_spider import BilibiliFavoritesSpider

class BilibiliRealTime:
    def __init__(self, cookies: str = None):
        """
        Initialize the BilibiliRealTime crawler

        Args:
            cookies: Optional cookies for authenticated requests
        """
        self.spider = BilibiliFavoritesSpider(cookies)
        self.user_map_path = "SSLRec/datasets/general_cf/bilibili/user_map.json"
        self.dataset_path = "dataset/bilibili"
        self.realtime_dataset_path = "dataset/bilibili_realtime"
        self.download_path = "download"

        # Load user mapping
        self.user_map = self._load_user_map()

    def __call__(self, user_id: str, save_data: bool = True, download_videos: bool = True, is_real_uid: bool = False) -> List[Dict]:
        """
        Get latest favorites for a specific user

        Args:
            user_id: The user ID (can be internal ID or real bilibili UID)
            save_data: Whether to automatically save the fetched data
            download_videos: Whether to download the actual video files
            is_real_uid: If True, user_id is treated as real bilibili UID; if False, as internal ID

        Returns:
            List of latest favorite videos (max 3)
        """
        if is_real_uid:
            # user_id is already a real bilibili UID
            real_uid = user_id
            # Try to get internal ID for data path compatibility
            internal_user_id = self.get_internal_user_id(real_uid)
            if not internal_user_id:
                print(f"警告: 用户 {real_uid} 在user_map中未找到对应的内部ID，使用real_uid作为标识")
                internal_user_id = real_uid
        else:
            # user_id is an internal ID, convert to real UID
            real_uid = self._get_real_uid(user_id)
            if not real_uid:
                print(f"用户ID {user_id} 未找到对应的真实B站UID")
                return []
            internal_user_id = user_id

        # Get latest timestamp from existing data
        latest_timestamp = self._get_latest_timestamp(internal_user_id)

        # Fetch latest favorites
        new_videos = self._fetch_latest_favorites(real_uid, internal_user_id, latest_timestamp)

        # Automatically save data if requested and data exists
        if save_data and new_videos:
            self._save_realtime_data(new_videos, internal_user_id, real_uid)

        # Download videos if requested and data exists
        if download_videos and new_videos:
            print(f"开始下载用户 {real_uid} 的实时视频...")
            asyncio.run(self._download_realtime_videos(new_videos, internal_user_id, real_uid, global_concurrent=False))

        return new_videos

    def _load_user_map(self) -> Dict[str, str]:
        """Load user mapping from JSON file"""
        try:
            with open(self.user_map_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"用户映射文件未找到: {self.user_map_path}")
            return {}
        except json.JSONDecodeError:
            print(f"用户映射文件格式错误: {self.user_map_path}")
            return {}

    def _get_real_uid(self, user_id: str) -> Optional[str]:
        """Convert internal user_id to real bilibili UID"""
        return self.user_map.get(user_id)

    def _get_latest_timestamp(self, user_id: str) -> int:
        """Get the latest fav_time timestamp from user's existing data"""
        user_dir = os.path.join(self.dataset_path, self.user_map.get(user_id, ""))
        if not os.path.exists(user_dir):
            print(f"用户数据目录不存在: {user_dir}")
            return 0

        latest_timestamp = 0
        try:
            # Find all CSV files in user directory
            csv_files = [f for f in os.listdir(user_dir) if f.endswith('.csv')]

            for csv_file in csv_files:
                csv_path = os.path.join(user_dir, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    if 'fav_time' in df.columns and not df.empty:
                        max_time = df['fav_time'].max()
                        if pd.notna(max_time):
                            latest_timestamp = max(latest_timestamp, int(max_time))
                except Exception as e:
                    print(f"读取CSV文件失败 {csv_path}: {e}")
                    continue

        except Exception as e:
            print(f"扫描用户目录失败 {user_dir}: {e}")

        print(f"用户 {user_id} 的最新时间戳: {latest_timestamp}")
        return latest_timestamp

    def _fetch_latest_favorites(self, real_uid: str, user_id: str, latest_timestamp: int) -> List[Dict]:
        """
        Fetch latest favorites that are newer than latest_timestamp

        Args:
            real_uid: Real bilibili UID
            user_id: Internal user ID
            latest_timestamp: Latest timestamp from existing data

        Returns:
            List of latest favorite videos (max 3)
        """
        try:
            # Get user's favorites list
            favorites = self.spider.get_user_favorites(real_uid)
            if not favorites:
                print(f"无法获取用户 {real_uid} 的收藏夹列表")
                return []

            print(f"获取到用户 {real_uid} 的 {len(favorites)} 个收藏夹")

            # Collect all new videos from all favorites
            new_videos = []

            for fav in favorites:
                if len(new_videos) >= 3:  # Already have enough videos
                    break

                fav_id = fav['id']
                fav_title = fav['title']

                print(f"检查收藏夹: {fav_title}")

                # Get contents of this favorite folder
                page = 1
                while len(new_videos) < 3:
                    videos = self.spider.get_favorite_contents(fav_id, fav_title, page)
                    if not videos:
                        break

                    for video in videos:
                        # Check if this video is newer than latest_timestamp
                        fav_time = video.get('fav_time', 0)
                        if isinstance(fav_time, (int, float)) and fav_time > latest_timestamp:
                            new_videos.append(video)
                            print(f"发现新视频: {video.get('title', 'Unknown')} (fav_time: {fav_time})")

                            if len(new_videos) >= 3:
                                break

                    # If no more videos or we have enough, break
                    if len(videos) < 20 or len(new_videos) >= 3:
                        break

                    page += 1
                    time.sleep(1)  # Be polite to the API

                time.sleep(2)  # Delay between favorites

            # Sort by fav_time descending and take top 3
            new_videos.sort(key=lambda x: x.get('fav_time', 0), reverse=True)
            result = new_videos[:3]

            print(f"用户 {real_uid} 找到 {len(result)} 个新收藏视频")
            return result

        except Exception as e:
            print(f"获取用户 {real_uid} 最新收藏失败: {e}")
            return []

    def _save_as_csv(self, videos: List[Dict], filepath: str):
        """
        Save videos data as CSV file

        Args:
            videos: List of video dictionaries
            filepath: Path to save the CSV file
        """
        if not videos:
            return

        # Get all possible fields
        fields = set()
        for video in videos:
            fields.update(video.keys())
        fields = sorted(fields)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Write CSV file
        with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(videos)

    def _save_realtime_data(self, videos: List[Dict], user_id: str, real_uid: str):
        """
        Save realtime data following bilibili_spider.py's structure

        Args:
            videos: List of video dictionaries to save
            user_id: Internal user ID
            real_uid: Real bilibili UID
        """
        if not videos:
            print(f"没有数据需要保存 (用户: {user_id})")
            return

        # Create timestamp for this batch
        timestamp = int(time.time())

        # Create user directory using real_uid
        user_dir = os.path.join(self.realtime_dataset_path, real_uid)
        os.makedirs(user_dir, exist_ok=True)

        # Group videos by favorite folder
        fav_contents = {}
        for video in videos:
            fav_id = video.get('fav_id', 'unknown')
            fav_title = video.get('fav_title', 'unknown')

            if fav_id not in fav_contents:
                fav_contents[fav_id] = {
                    'title': fav_title,
                    'videos': []
                }
            fav_contents[fav_id]['videos'].append(video)

        # Save each favorite folder as separate CSV file
        saved_files = []
        for fav_id, content in fav_contents.items():
            fav_title = content['title']
            fav_videos = content['videos']

            # Create safe filename (same logic as bilibili_spider.py)
            safe_title = "".join([c for c in fav_title if c.isalnum() or c in ' _-'])
            safe_title = safe_title[:50]  # Limit length

            # Create filename with timestamp to avoid conflicts
            filename = f"{safe_title}_{fav_id}_{timestamp}.csv"
            filepath = os.path.join(user_dir, filename)

            # Save CSV file
            self._save_as_csv(fav_videos, filepath)
            saved_files.append(filepath)

            print(f"保存收藏夹数据: {filepath} ({len(fav_videos)} 个视频)")

        # Create summary file for this batch
        if len(fav_contents) > 1:
            summary_filename = f"realtime_summary_{timestamp}.csv"
            summary_filepath = os.path.join(user_dir, summary_filename)
            self._save_as_csv(videos, summary_filepath)
            saved_files.append(summary_filepath)
            print(f"保存汇总数据: {summary_filepath}")

        print(f"用户 {real_uid} 实时数据保存完成，共保存 {len(saved_files)} 个文件")
        return saved_files

    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a user"""
        real_uid = self._get_real_uid(user_id)
        if not real_uid:
            return {"error": "用户ID未找到"}

        # Count existing videos
        user_dir = os.path.join(self.dataset_path, real_uid)
        total_videos = 0

        if os.path.exists(user_dir):
            for file in os.listdir(user_dir):
                if file.endswith('.csv'):
                    csv_path = os.path.join(user_dir, file)
                    try:
                        df = pd.read_csv(csv_path)
                        total_videos += len(df)
                    except Exception:
                        continue

        return {
            "user_id": user_id,
            "real_uid": real_uid,
            "total_videos": total_videos,
            "dataset_path": user_dir
        }

    async def _download_realtime_videos(self, videos: List[Dict], user_id: str, real_uid: str, global_concurrent: bool = False, max_concurrent_downloads: int = 10):
        """
        Download realtime videos to download/{user}/realtime directory

        Args:
            videos: List of video dictionaries to download
            user_id: Internal user ID
            real_uid: Real bilibili UID
            global_concurrent: Whether to use global concurrency (for single user, this is same as batch mode)
            max_concurrent_downloads: Maximum concurrent downloads
        """
        if not videos:
            print(f"没有视频需要下载 (用户: {user_id})")
            return

        # Check if yt-dlp is available
        if not self._ensure_ytdlp_available():
            print("❌ yt-dlp 不可用，跳过视频下载")
            return

        # Create realtime download directory
        realtime_dir = os.path.join(self.download_path, "bilibili", real_uid, "realtime")
        os.makedirs(realtime_dir, exist_ok=True)

        print(f"实时视频下载目录: {realtime_dir}")
        print(f"准备下载 {len(videos)} 个实时视频...")
        print(f"并发模式: {'全局并发' if global_concurrent else '批次并发'}")

        if global_concurrent:
            # Use global concurrent mode for single user
            await self._global_concurrent_download_single_user(videos, real_uid, realtime_dir, max_concurrent_downloads)
        else:
            # Use original batch mode
            await self._batch_download_single_user(videos, real_uid, realtime_dir)

    async def _global_concurrent_download_single_user(self, videos: List[Dict], real_uid: str, realtime_dir: str, max_concurrent_downloads: int):
        """
        Global concurrent download for single user's videos
        """
        print(f"🚀 启用全局并发下载模式 (最大并发: {max_concurrent_downloads})...")

        # Create download tasks
        download_tasks = []
        video_infos = []

        for video in videos:
            bvid = video.get('bvid')
            if bvid:
                # Create individual folder for each video
                video_folder = os.path.join(realtime_dir, bvid)
                os.makedirs(video_folder, exist_ok=True)

                video_info = {
                    'bvid': bvid,
                    'video_folder': video_folder,
                    'real_uid': real_uid,
                    'title': video.get('title', bvid)[:50]
                }

                task = self._download_single_video(bvid, video_folder, video)
                download_tasks.append(task)
                video_infos.append(video_info)
            else:
                print(f"⚠️ 警告: 视频缺少bvid信息: {video}")

        if not download_tasks:
            print("❌ 没有有效的视频可以下载")
            return

        # Use semaphore to limit global concurrency
        semaphore = asyncio.Semaphore(max_concurrent_downloads)
        successful_downloads = 0
        failed_downloads = 0

        async def limited_download(video_info, task):
            async with semaphore:
                try:
                    result = await task
                    if result:
                        print(f"✅ {video_info['bvid']} 下载成功: {video_info['title']}")
                        return True
                    else:
                        print(f"❌ {video_info['bvid']} 下载失败: {video_info['title']}")
                        return False
                except Exception as e:
                    print(f"❌ {video_info['bvid']} 下载异常: {e}")
                    return False

        # Create limited tasks
        limited_tasks = []
        for video_info, task in zip(video_infos, download_tasks):
            limited_task = limited_download(video_info, task)
            limited_tasks.append(limited_task)

        # Execute all downloads concurrently
        start_time = time.time()
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        end_time = time.time()

        # Count results
        for result in results:
            if isinstance(result, Exception):
                failed_downloads += 1
                print(f"❌ 下载异常: {result}")
            elif result:
                successful_downloads += 1
            else:
                failed_downloads += 1

        # Print summary
        print(f"\n🎉 全局并发下载完成:")
        print(f"  ✅ 成功下载: {successful_downloads}/{len(download_tasks)} 个视频")
        print(f"  ❌ 失败下载: {failed_downloads}/{len(download_tasks)} 个视频")
        print(f"  ⏱️ 总耗时: {end_time - start_time:.2f} 秒")
        if end_time - start_time > 0:
            print(f"  🚀 平均速度: {len(download_tasks)/(end_time - start_time):.2f} 任务/秒")
        print(f"  👤 用户: {real_uid}")
        print(f"  📁 下载目录: {realtime_dir}")

    async def _batch_download_single_user(self, videos: List[Dict], real_uid: str, realtime_dir: str):
        """
        Original batch download for single user's videos
        """
        print(f"📱 使用批次并发下载模式...")

        # Create download tasks
        download_tasks = []
        for video in videos:
            bvid = video.get('bvid')
            if bvid:
                # Create individual folder for each video
                video_folder = os.path.join(realtime_dir, bvid)
                os.makedirs(video_folder, exist_ok=True)

                task = self._download_single_video(bvid, video_folder, video)
                download_tasks.append(task)
            else:
                print(f"⚠️ 警告: 视频缺少bvid信息: {video}")

        if not download_tasks:
            print("❌ 没有有效的视频可以下载")
            return

        # Execute downloads with limited concurrency
        successful_downloads = 0
        batch_size = 3  # Download 3 videos concurrently to avoid rate limiting

        for batch_idx in range(0, len(download_tasks), batch_size):
            batch_tasks = download_tasks[batch_idx:batch_idx + batch_size]
            print(f"📦 处理下载批次 {batch_idx//batch_size + 1}/{(len(download_tasks) + batch_size - 1)//batch_size} ({len(batch_tasks)} 个视频)")

            # Add delay between batches (except for the first batch)
            if batch_idx > 0:
                delay = random.uniform(2, 3)  # 2-3 seconds between batches
                print(f"⏳ 等待 {delay:.1f} 秒后开始下一批次...")
                await asyncio.sleep(delay)

            # Execute current batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Count successful downloads
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"❌ 下载异常: {result}")
                elif result:
                    successful_downloads += 1

            print(f"✅ 批次完成: {sum(1 for r in batch_results if r is True)}/{len(batch_results)} 个视频下载成功")

        print(f"\n📊 实时视频下载总结:")
        print(f"  ✅ 成功下载: {successful_downloads}/{len(download_tasks)} 个视频")
        print(f"  👤 用户: {real_uid}")
        print(f"  📁 下载目录: {realtime_dir}")

    async def _download_single_video(self, bvid: str, save_path: str, video_info: Dict, max_retries: int = 3, max_duration_hours: int = 0.1, max_videos_per_playlist: int = 1) -> bool:
        """
        Download a single bilibili video with retry mechanism

        Args:
            bvid: Bilibili video ID
            save_path: Path to save the video
            video_info: Video information dictionary
            max_retries: Maximum retry attempts
            max_duration_hours: Maximum video duration in hours
            max_videos_per_playlist: Maximum videos to download from playlists

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid being detected as bot
                if attempt > 0:
                    delay = random.uniform(5, 10)  # 5-10 seconds delay
                    print(f"重试下载 {bvid}, 等待 {delay:.1f} 秒... (尝试 {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)

                url = f"https://www.bilibili.com/video/{bvid}"
                video_title = video_info.get('title', bvid)[:50]  # Limit title length for display
                print(f"下载实时视频: {video_title} ({bvid}) 到 {save_path}")

                # Enhanced yt-dlp command with better options and duration limit
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
                    print(f"✅ 实时视频下载成功: {bvid}")
                    return True
                else:
                    try:
                        error_msg = stderr.decode('utf-8')
                    except UnicodeDecodeError:
                        error_msg = stderr.decode('gbk', errors='ignore')

                    # Check if it's a permanent error (no need to retry)
                    if any(keyword in error_msg.lower() for keyword in ['private', 'deleted', 'not available', 'geo-blocked']):
                        print(f"❌ 永久性错误，跳过 {bvid}: {error_msg}")
                        return False

                    print(f"⚠️ 下载尝试 {attempt + 1} 失败 {bvid}: {error_msg}")

                    if attempt == max_retries - 1:
                        print(f"❌ 所有下载尝试都失败 {bvid}")
                        return False

            except Exception as e:
                print(f"❌ 下载异常 {bvid}: {e}")
                if attempt == max_retries - 1:
                    return False

        return False

    def _check_ytdlp_available(self) -> bool:
        """
        Check if yt-dlp is available in the system

        Returns:
            True if yt-dlp is available, False otherwise
        """
        return shutil.which("yt-dlp") is not None

    def _install_ytdlp(self) -> bool:
        """
        Try to install yt-dlp using pip

        Returns:
            True if installation successful, False otherwise
        """
        try:
            print("🔧 yt-dlp 未找到，尝试安装...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"],
                                  capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("✅ yt-dlp 安装成功")
                return True
            else:
                print(f"❌ yt-dlp 安装失败: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ 安装 yt-dlp 时出错: {e}")
            return False

    def _ensure_ytdlp_available(self) -> bool:
        """
        Ensure yt-dlp is available, install if necessary

        Returns:
            True if yt-dlp is available, False otherwise
        """
        if self._check_ytdlp_available():
            return True

        print("⚠️ 检测到 yt-dlp 未安装，视频下载功能需要 yt-dlp")

        # Ask user if they want to install
        try:
            choice = input("是否要自动安装 yt-dlp? (y/n): ").lower().strip()
            if choice == 'y':
                return self._install_ytdlp()
            else:
                print("❌ 用户选择不安装 yt-dlp，跳过视频下载")
                return False
        except:
            # If input fails (e.g., in non-interactive environment), try to install automatically
            print("🤖 非交互环境，尝试自动安装 yt-dlp...")
            return self._install_ytdlp()

    def get_all_download_users(self) -> List[str]:
        """
        Get all user IDs that exist in the download/bilibili directory

        Returns:
            List of real user IDs (not internal IDs)
        """
        bilibili_download_dir = os.path.join(self.download_path, "bilibili")

        if not os.path.exists(bilibili_download_dir):
            print(f"Bilibili下载目录不存在: {bilibili_download_dir}")
            return []

        # Get all subdirectories (these are real user IDs)
        user_dirs = []
        try:
            for item in os.listdir(bilibili_download_dir):
                item_path = os.path.join(bilibili_download_dir, item)
                if os.path.isdir(item_path):
                    user_dirs.append(item)

            print(f"发现 {len(user_dirs)} 个用户目录: {user_dirs}")
            return user_dirs

        except Exception as e:
            print(f"扫描下载目录失败: {e}")
            return []

    def get_internal_user_id(self, real_uid: str) -> Optional[str]:
        """
        Convert real bilibili UID to internal user_id (reverse of _get_real_uid)

        Args:
            real_uid: Real bilibili UID

        Returns:
            Internal user ID if found, None otherwise
        """
        for internal_id, real_id in self.user_map.items():
            if real_id == real_uid:
                return internal_id
        return None

    def process_all_users(self, save_data: bool = True, download_videos: bool = True, global_concurrent: bool = True, max_concurrent_downloads: int = 15) -> Dict[str, List[Dict]]:
        """
        Process realtime data for all users in the download directory

        Args:
            save_data: Whether to save the fetched data
            download_videos: Whether to download the actual video files
            global_concurrent: Whether to use global concurrency across all users
            max_concurrent_downloads: Maximum concurrent downloads globally

        Returns:
            Dictionary mapping user_id to list of new videos
        """
        print("🚀 开始处理所有用户的实时数据...")
        print(f"🔧 并发模式: {'全局并发' if global_concurrent else '按用户串行'}")
        if global_concurrent and download_videos:
            print(f"🎯 最大并发下载数: {max_concurrent_downloads}")
        print("=" * 60)

        # Get all users from download directory
        real_user_ids = self.get_all_download_users()

        if not real_user_ids:
            print("❌ 没有找到任何用户目录")
            return {}

        if global_concurrent and download_videos:
            # Use global concurrent mode
            return asyncio.run(self._process_all_users_global_concurrent(real_user_ids, save_data, max_concurrent_downloads))
        else:
            # Use original sequential mode
            return self._process_all_users_sequential(real_user_ids, save_data, download_videos)

    def _process_all_users_sequential(self, real_user_ids: List[str], save_data: bool, download_videos: bool) -> Dict[str, List[Dict]]:
        """
        Original sequential processing of all users
        """
        print("📱 使用按用户串行处理模式...")

        results = {}
        total_new_videos = 0

        for i, real_uid in enumerate(real_user_ids, 1):
            print(f"\n📋 处理用户 {i}/{len(real_user_ids)}: {real_uid}")
            print("-" * 40)

            try:
                # Process this user's realtime data using real UID directly
                new_videos = self(real_uid, save_data=save_data, download_videos=download_videos, is_real_uid=True)
                results[real_uid] = new_videos
                total_new_videos += len(new_videos)

                print(f"✅ 用户 {real_uid} 处理完成: {len(new_videos)} 个新视频")

                # Add delay between users to be polite
                if i < len(real_user_ids):  # Don't delay after the last user
                    delay = random.uniform(2, 4)
                    print(f"⏳ 等待 {delay:.1f} 秒后处理下一个用户...")
                    time.sleep(delay)

            except Exception as e:
                print(f"❌ 处理用户 {real_uid} 时出错: {e}")
                results[real_uid] = []
                continue

        print("\n" + "=" * 60)
        print("🎉 所有用户处理完成！")
        print(f"📊 总结:")
        print(f"  - 处理用户数: {len(real_user_ids)}")
        print(f"  - 总新视频数: {total_new_videos}")
        print(f"  - 成功处理: {len([r for r in results.values() if r])} 个用户有新内容")
        print("=" * 60)

        return results

    async def _process_all_users_global_concurrent(self, real_user_ids: List[str], save_data: bool, max_concurrent_downloads: int) -> Dict[str, List[Dict]]:
        """
        Global concurrent processing of all users with cross-user video download concurrency
        """
        print("🚀 使用全局并发处理模式...")

        # Step 1: Collect all new videos from all users (without downloading)
        print("\n📊 第一阶段: 收集所有用户的新视频数据...")
        all_user_videos = {}
        total_videos = 0

        for i, real_uid in enumerate(real_user_ids, 1):
            print(f"📋 收集用户 {i}/{len(real_user_ids)}: {real_uid}")

            try:
                # Get new videos without downloading
                new_videos = self(real_uid, save_data=save_data, download_videos=False, is_real_uid=True)
                all_user_videos[real_uid] = new_videos
                total_videos += len(new_videos)

                print(f"✅ 用户 {real_uid}: 发现 {len(new_videos)} 个新视频")

                # Small delay between API calls
                if i < len(real_user_ids):
                    await asyncio.sleep(1)

            except Exception as e:
                print(f"❌ 收集用户 {real_uid} 数据时出错: {e}")
                all_user_videos[real_uid] = []
                continue

        print(f"\n📈 数据收集完成:")
        print(f"  📊 总用户数: {len(real_user_ids)}")
        print(f"  🎬 总视频数: {total_videos}")
        print(f"  ✅ 有新内容的用户: {len([v for v in all_user_videos.values() if v])}")

        if total_videos == 0:
            print("❌ 没有发现新视频，跳过下载阶段")
            return all_user_videos

        # Step 2: Global concurrent download across all users
        print(f"\n🎯 第二阶段: 全局并发下载所有视频 (最大并发: {max_concurrent_downloads})...")

        # Check if yt-dlp is available
        if not self._ensure_ytdlp_available():
            print("❌ yt-dlp 不可用，跳过视频下载")
            return all_user_videos

        # Create all download tasks
        all_download_tasks = []
        video_task_mapping = []  # Track which task belongs to which user/video

        for real_uid, videos in all_user_videos.items():
            if not videos:
                continue

            # Create realtime download directory for this user
            realtime_dir = os.path.join(self.download_path, "bilibili", real_uid, "realtime")
            os.makedirs(realtime_dir, exist_ok=True)

            for video in videos:
                bvid = video.get('bvid')
                if bvid:
                    # Create individual folder for each video
                    video_folder = os.path.join(realtime_dir, bvid)
                    os.makedirs(video_folder, exist_ok=True)

                    task_info = {
                        'bvid': bvid,
                        'real_uid': real_uid,
                        'video_folder': video_folder,
                        'title': video.get('title', bvid)[:50]
                    }

                    task = self._download_single_video(bvid, video_folder, video)
                    all_download_tasks.append(task)
                    video_task_mapping.append(task_info)

        if not all_download_tasks:
            print("❌ 没有有效的视频可以下载")
            return all_user_videos

        print(f"📦 准备下载 {len(all_download_tasks)} 个视频...")

        # Use semaphore to limit global concurrency
        semaphore = asyncio.Semaphore(max_concurrent_downloads)
        successful_downloads = 0
        failed_downloads = 0
        user_success_count = {uid: 0 for uid in all_user_videos.keys()}

        async def limited_download(task_info, task):
            async with semaphore:
                try:
                    result = await task
                    if result:
                        user_success_count[task_info['real_uid']] += 1
                        print(f"✅ {task_info['bvid']} 下载成功 ({task_info['real_uid']}): {task_info['title']}")
                        return True
                    else:
                        print(f"❌ {task_info['bvid']} 下载失败 ({task_info['real_uid']}): {task_info['title']}")
                        return False
                except Exception as e:
                    print(f"❌ {task_info['bvid']} 下载异常 ({task_info['real_uid']}): {e}")
                    return False

        # Create limited tasks
        limited_tasks = []
        for task_info, task in zip(video_task_mapping, all_download_tasks):
            limited_task = limited_download(task_info, task)
            limited_tasks.append(limited_task)

        # Execute all downloads concurrently
        start_time = time.time()
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        end_time = time.time()

        # Count results
        for result in results:
            if isinstance(result, Exception):
                failed_downloads += 1
                print(f"❌ 下载异常: {result}")
            elif result:
                successful_downloads += 1
            else:
                failed_downloads += 1

        # Print summary
        print(f"\n🎉 全局并发下载完成:")
        print(f"  ✅ 成功下载: {successful_downloads}/{len(all_download_tasks)} 个视频")
        print(f"  ❌ 失败下载: {failed_downloads}/{len(all_download_tasks)} 个视频")
        print(f"  ⏱️ 总耗时: {end_time - start_time:.2f} 秒")
        if end_time - start_time > 0:
            print(f"  🚀 平均速度: {len(all_download_tasks)/(end_time - start_time):.2f} 任务/秒")

        # Print per-user summary
        print(f"\n👥 各用户下载统计:")
        for real_uid, videos in all_user_videos.items():
            if videos:
                success_count = user_success_count[real_uid]
                total_count = len(videos)
                print(f"  📋 {real_uid}: {success_count}/{total_count} 成功")

        print("=" * 60)

        return all_user_videos
