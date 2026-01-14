import os
import pandas as pd
import time
import csv
import asyncio
import random
import requests
from bs4 import BeautifulSoup
import re
import html
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor


class HupuRealTime:
    def __init__(self, cookies: str = None):
        """
        Initialize the HupuRealTime crawler

        Args:
            cookies: Optional cookies for authenticated requests
        """
        self.cookies = cookies
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Cookie": cookies or "",
            "Referer": "https://bbs.hupu.com/",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Connection": "keep-alive"
        }

        # Paths configuration
        self.dataset_path = "dataset/hupu"
        self.realtime_dataset_path = "dataset/hupu_realtime"
        self.download_path = "download"

    def __call__(self, user_id: str, save_data: bool = True, download_media: bool = True) -> List[Dict]:
        """
        Get latest replies for a specific user

        Args:
            user_id: The hupu user ID
            save_data: Whether to automatically save the fetched data
            download_media: Whether to download the actual media files

        Returns:
            List of latest replies (max 3)
        """
        print(f"处理虎扑用户: {user_id}")

        # Get latest timestamp from existing data
        latest_timestamp = self._get_latest_timestamp(user_id)

        # Fetch latest replies
        new_posts = self._fetch_latest_replies(user_id, latest_timestamp)

        # Step 1: Save data to CSV first
        csv_filepath = None
        if save_data and new_posts:
            csv_filepath = self._save_realtime_data(new_posts, user_id)

        # Step 2: Download media files by reading from the saved CSV
        if download_media and csv_filepath:
            print(f"开始从 CSV 读取数据并下载用户 {user_id} 的实时媒体文件...")
            # Load posts from CSV file
            posts_from_csv = self._load_posts_from_csv(csv_filepath[0] if isinstance(csv_filepath, list) else csv_filepath)
            if posts_from_csv:
                asyncio.run(self._download_realtime_media(posts_from_csv, user_id))
            else:
                print(f"⚠️ 无法从 CSV 加载数据: {csv_filepath}")

        return new_posts

    def _load_posts_from_csv(self, csv_filepath: str) -> List[Dict]:
        """
        Load posts data from a CSV file

        Args:
            csv_filepath: Path to the CSV file

        Returns:
            List of post dictionaries
        """
        if not os.path.exists(csv_filepath):
            print(f"CSV 文件不存在: {csv_filepath}")
            return []

        try:
            posts = []
            df = pd.read_csv(csv_filepath)

            for _, row in df.iterrows():
                post = {
                    'user_id': str(row.get('user_id', '')),
                    'title': str(row.get('title', '')),
                    'reply': str(row.get('reply', '')),
                    'tag': str(row.get('tag', '')),
                    'fav_time': str(row.get('fav_time', '')),
                    'images': str(row.get('images', '')) if pd.notna(row.get('images')) else '',
                    'pid': str(row.get('pid', '')),
                    'tid': str(row.get('hupuID', '')),  # Map hupuID back to tid
                    'videos': str(row.get('videos', '')) if pd.notna(row.get('videos')) else '',
                    'text': str(row.get('text', '')) if pd.notna(row.get('text')) else ''
                }
                posts.append(post)

            print(f"✅ 从 CSV 加载了 {len(posts)} 条数据: {csv_filepath}")
            return posts

        except Exception as e:
            print(f"❌ 读取 CSV 文件失败 {csv_filepath}: {e}")
            return []

    def _get_latest_timestamp(self, user_id: str) -> int:
        """Get the latest fav_time timestamp from user's existing data"""
        user_dir = os.path.join(self.dataset_path, user_id)

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
                        for fav_time in df['fav_time']:
                            if pd.notna(fav_time):
                                # Convert YYYYMMDDHHMMSS format to timestamp
                                timestamp = self._convert_to_timestamp(str(fav_time))
                                if timestamp:
                                    latest_timestamp = max(latest_timestamp, timestamp)
                except Exception as e:
                    print(f"读取CSV文件失败 {csv_path}: {e}")
                    continue

        except Exception as e:
            print(f"扫描用户目录失败 {user_dir}: {e}")

        print(f"用户 {user_id} 的最新时间戳: {latest_timestamp}")
        return latest_timestamp

    def _convert_to_timestamp(self, time_value) -> int:
        """Convert various time formats to timestamp"""
        if isinstance(time_value, (int, float)):
            # Check if it's already a unix timestamp or YYYYMMDDHHMMSS format
            if time_value > 20000000000000:  # It's in YYYYMMDDHHMMSS format
                return self._parse_hupu_time(str(int(time_value)))
            return int(time_value)

        if isinstance(time_value, str):
            try:
                # Try to parse as YYYYMMDDHHMMSS format
                if len(time_value) == 14 and time_value.isdigit():
                    return self._parse_hupu_time(time_value)
                # Try to parse as timestamp
                if time_value.isdigit():
                    return int(time_value)
            except:
                return 0

        return 0

    def _parse_hupu_time(self, time_str: str) -> int:
        """Parse hupu time format YYYYMMDDHHMMSS to unix timestamp"""
        try:
            from datetime import datetime
            dt = datetime.strptime(time_str, '%Y%m%d%H%M%S')
            return int(dt.timestamp())
        except:
            return 0

    def _format_time_for_csv(self, timestamp: int) -> str:
        """Format unix timestamp to YYYYMMDDHHMMSS for CSV storage"""
        try:
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime('%Y%m%d%H%M%S')
        except:
            return ''

    def _clean_text_fast(self, text: str) -> str:
        """Fast clean text: remove HTML tags, whitespace, decode HTML entities"""
        if not text:
            return ''
        text = html.unescape(text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _normalize(self, text: str) -> str:
        """Remove spaces, newlines, zero-width characters for matching"""
        return re.sub(r'\s+', '', text)

    def _fetch_post_text(self, tid: str) -> str:
        """Fetch post content text"""
        if not tid:
            return ''
        try:
            url = f"https://bbs.hupu.com/{tid}.html"
            resp = requests.get(url, headers=self.headers, timeout=5)
            resp.raise_for_status()
            time.sleep(random.uniform(0.5, 1.0))
            soup = BeautifulSoup(resp.text, 'html.parser')

            content_div = soup.select_one('div.thread-content-detail')
            if not content_div:
                return ''

            for img_div in content_div.select('div[data-hupu-node="image"]'):
                img_div.decompose()

            text = content_div.get_text(separator='\n', strip=True)
            return self._clean_text_fast(text)

        except Exception as e:
            print(f"⚠️ tid {tid} 获取正文失败: {e}")
            return ''

    def _fetch_post_images(self, tid: str) -> str:
        """Fetch post image URLs"""
        if not tid:
            return ''
        try:
            url = f"https://bbs.hupu.com/{tid}.html"
            resp = requests.get(url, headers=self.headers, timeout=5)
            resp.raise_for_status()
            time.sleep(random.uniform(0.5, 1.0))
            soup = BeautifulSoup(resp.text, 'html.parser')

            images = set()
            filtered_url = "https://w1.hoopchina.com.cn/games/images/def_man.png"

            # Extract images from div[data-hupu-node="image"]
            for div in soup.find_all('div', attrs={'data-hupu-node': 'image'}):
                for img in div.find_all('img'):
                    if img.get('src'):
                        images.add(img['src'])
                    if img.get('data-origin'):
                        images.add(img['data-origin'])

            # Extract from regular <img> tags
            for img in soup.find_all('img'):
                if img.get('data-origin'):
                    images.add(img['data-origin'])

            # Filter out default avatar image
            images.discard(filtered_url)

            return ';'.join(images)
        except Exception as e:
            print(f"⚠️ tid {tid} 获取图片失败: {e}")
            return ''

    def _fetch_post_videos(self, tid: str) -> str:
        """Fetch post video URLs"""
        if not tid:
            return ''
        try:
            url = f"https://bbs.hupu.com/{tid}.html"
            resp = requests.get(url, headers=self.headers, timeout=5)
            resp.raise_for_status()
            time.sleep(random.uniform(0.5, 1.0))
            soup = BeautifulSoup(resp.text, 'html.parser')

            videos = [v.get('src') for v in soup.find_all('video') if v.get('src')]
            return ';'.join(videos)
        except Exception as e:
            print(f"⚠️ tid {tid} 获取视频失败: {e}")
            return ''

    def _fetch_post_data_multithread(self, tid_list: List[str], max_workers: int = 5) -> Dict:
        """Fetch post data (text, images, videos) concurrently"""
        result_dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for tid in tid_list:
                futures[executor.submit(self._fetch_post_images, tid)] = (tid, 'images')
                futures[executor.submit(self._fetch_post_text, tid)] = (tid, 'text')
                futures[executor.submit(self._fetch_post_videos, tid)] = (tid, 'videos')

            for future in futures:
                tid, data_type = futures[future]
                try:
                    value = future.result()
                    if tid not in result_dict:
                        result_dict[tid] = {'images': '', 'text': '', 'videos': ''}
                    result_dict[tid][data_type] = value
                except Exception as e:
                    print(f"⚠️ tid {tid} 并发抓取 {data_type} 失败: {e}")
                    if tid not in result_dict:
                        result_dict[tid] = {'images': '', 'text': '', 'videos': ''}
                time.sleep(random.uniform(0.2, 0.5))
        return result_dict

    def _fetch_latest_replies(self, user_id: str, latest_timestamp: int) -> List[Dict]:
        """
        Fetch latest replies that are newer than latest_timestamp

        Args:
            user_id: Hupu user ID
            latest_timestamp: Latest timestamp from existing data

        Returns:
            List of latest replies (max 3)
        """
        try:
            url = f"https://my.hupu.com/{user_id}?tabKey=2"
            resp = requests.get(url, headers=self.headers, timeout=5)
            resp.raise_for_status()
            time.sleep(random.uniform(0.5, 1.0))
            soup = BeautifulSoup(resp.text, 'html.parser')
            html_str = str(soup)

            # Extract pid, tid, content from JSON in HTML
            pattern = re.compile(r'"pid":(\d+).*?"tid":(\d+).*?"content":"(.*?)"', re.S)
            matches = pattern.findall(html_str)
            pid_tid_content_list = [
                {'pid': pid, 'tid': tid, 'content': self._clean_text_fast(content)}
                for pid, tid, content in matches
            ]

            posts = []
            tid_list = []

            # Keywords to filter out deleted/hidden posts
            skip_keywords = ['主帖已被隐藏', '主帖已被删除']

            for item in soup.select('div.list-item'):
                reply_tag = item.select_one('div.list-item-reply')
                main_post_tag = item.select_one('span.shoImgWarp > a')
                topic_tag = item.select_one('span.hasImgContentTime > span > a.hasTopicName')
                time_tag = item.select_one('span.hasImgContentTime')

                # Get title and check if it should be skipped
                title = main_post_tag.get_text(strip=True) if main_post_tag else ''
                if any(keyword in title for keyword in skip_keywords):
                    print(f"⏭️ 跳过已删除/隐藏的帖子: {title}")
                    continue

                reply_text = self._clean_text_fast(reply_tag.get_text() if reply_tag else '')

                pid_val, tid_val = '', ''
                for entry in pid_tid_content_list:
                    if self._normalize(entry['content']) == self._normalize(reply_text):
                        pid_val = entry['pid']
                        tid_val = entry['tid']
                        tid_list.append(tid_val)
                        break

                time_str = ''
                unix_timestamp = 0
                if time_tag:
                    raw_time = time_tag.get_text(strip=True)
                    if topic_tag:
                        raw_time = raw_time.replace('来自：' + topic_tag.get_text(strip=True), '')
                    match = re.search(r'(\d{4})-(\d{2})-(\d{2})\s*(\d{2}):(\d{2}):(\d{2})', raw_time)
                    if match:
                        time_str = ''.join(match.groups())
                        unix_timestamp = self._parse_hupu_time(time_str)

                posts.append({
                    'user_id': user_id,
                    'pid': pid_val,
                    'tid': tid_val,
                    'title': title,
                    'reply': reply_text,
                    'tag': topic_tag.get_text(strip=True) if topic_tag else '',
                    'fav_time': time_str,
                    'unix_timestamp': unix_timestamp,
                    'images': '',
                    'videos': '',
                    'text': ''
                })

            # Filter posts newer than latest_timestamp
            new_posts = []
            for post in posts:
                if post['unix_timestamp'] > latest_timestamp:
                    new_posts.append(post)
                    print(f"发现新回帖: {post['title'][:50]}... (时间: {post['fav_time']})")

            # Sort by time descending and take top 3
            new_posts.sort(key=lambda x: x['unix_timestamp'], reverse=True)
            new_posts = new_posts[:3]

            # Fetch additional data for new posts
            if new_posts:
                tid_list = [p['tid'] for p in new_posts if p['tid']]
                if tid_list:
                    post_data_dict = self._fetch_post_data_multithread(tid_list)
                    for post in new_posts:
                        if post['tid'] in post_data_dict:
                            post['images'] = post_data_dict[post['tid']]['images']
                            post['videos'] = post_data_dict[post['tid']]['videos']
                            post['text'] = post_data_dict[post['tid']]['text']

            print(f"用户 {user_id} 找到 {len(new_posts)} 条新回帖")
            return new_posts

        except Exception as e:
            print(f"⚠️ 用户 {user_id} 抓取回帖失败: {e}")
            return []

    def _save_realtime_data(self, posts: List[Dict], user_id: str):
        """
        Save realtime data as CSV following hupu structure

        Args:
            posts: List of post dictionaries to save
            user_id: Hupu user ID
        """
        if not posts:
            print(f"没有数据需要保存 (用户: {user_id})")
            return

        # Create timestamp for this batch
        timestamp = int(time.time())

        # Create user directory
        user_dir = os.path.join(self.realtime_dataset_path, user_id)
        os.makedirs(user_dir, exist_ok=True)

        # Create CSV filename
        csv_filename = f"realtime_{timestamp}.csv"
        csv_filepath = os.path.join(user_dir, csv_filename)

        # Save as CSV file
        self._save_as_csv(posts, csv_filepath)

        print(f"用户 {user_id} 实时数据保存完成: {csv_filepath} ({len(posts)} 条回帖)")
        return [csv_filepath]

    def _save_as_csv(self, posts: List[Dict], filepath: str):
        """
        Save posts data as CSV file matching hupu format

        Args:
            posts: List of post dictionaries
            filepath: Path to save the CSV file
        """
        if not posts:
            return

        # Define CSV columns matching the hupu format
        fieldnames = [
            'user_id', 'title', 'reply', 'tag', 'fav_time',
            'images', 'pid', 'hupuID', 'videos', 'text'
        ]

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Write CSV file
        with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for post in posts:
                csv_row = {
                    'user_id': post.get('user_id', ''),
                    'title': post.get('title', ''),
                    'reply': post.get('reply', ''),
                    'tag': post.get('tag', ''),
                    'fav_time': post.get('fav_time', ''),
                    'images': post.get('images', ''),
                    'pid': post.get('pid', ''),
                    'hupuID': post.get('tid', ''),
                    'videos': post.get('videos', ''),
                    'text': post.get('text', '')
                }
                writer.writerow(csv_row)

    async def _download_realtime_media(self, posts: List[Dict], user_id: str):
        """
        Download realtime media files to download/{user}/realtime directory

        Args:
            posts: List of post dictionaries to download
            user_id: Hupu user ID
        """
        if not posts:
            print(f"没有媒体需要下载 (用户: {user_id})")
            return

        # Create realtime download directory
        realtime_dir = os.path.join(self.download_path, "hupu", user_id, "realtime")
        os.makedirs(realtime_dir, exist_ok=True)

        print(f"实时媒体下载目录: {realtime_dir}")
        print(f"准备下载 {len(posts)} 个帖子的媒体文件...")

        successful_downloads = 0

        for i, post in enumerate(posts, 1):
            tid = post.get('tid', f'post_{i}')
            title = post.get('title', '无标题')

            print(f"[{i}/{len(posts)}] 处理帖子: {title[:50]}...")

            # Create individual folder for each post
            post_folder = os.path.join(realtime_dir, str(tid))
            os.makedirs(post_folder, exist_ok=True)

            try:
                # Save text content
                text_success = self._save_text_content(post, post_folder)

                # Download images
                images_success = await self._download_post_images(post, post_folder)

                # Download videos
                videos_success = await self._download_post_videos(post, post_folder)

                if text_success or images_success or videos_success:
                    successful_downloads += 1
                    print(f"✅ 帖子 {tid} 媒体下载完成")

            except Exception as e:
                print(f"❌ 帖子 {tid} 媒体下载失败: {e}")

            # Add delay between posts
            if i < len(posts):
                await asyncio.sleep(random.uniform(1, 2))

        print(f"\n📊 实时媒体下载总结:")
        print(f"  ✅ 成功下载: {successful_downloads}/{len(posts)} 个帖子")
        print(f"  👤 用户: {user_id}")
        print(f"  📁 下载目录: {realtime_dir}")

    def _save_text_content(self, post: Dict, save_path: str) -> bool:
        """
        Save post content as txt file

        Args:
            post: Post dictionary
            save_path: Directory to save the text file

        Returns:
            True if successful, False otherwise
        """
        try:
            title = post.get('title', '无标题')
            reply = post.get('reply', '')
            text = post.get('text', '')

            # Combine content
            content = f"标题: {title}\n\n回帖: {reply}\n\n正文: {text}"

            # Sanitize filename
            clean_title = self._sanitize_filename(title)
            filename = f"{clean_title}.txt"
            filepath = os.path.join(save_path, filename)

            # Write content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"        ✅ 文本保存成功: {filename}")
            return True

        except Exception as e:
            print(f"        ❌ 文本保存失败: {e}")
            return False

    def _sanitize_filename(self, filename: str) -> str:
        """
        Clean filename of invalid characters

        Args:
            filename: Original filename

        Returns:
            Sanitized filename safe for filesystem
        """
        if not filename or filename.strip() == '':
            return "无标题"

        # Remove invisible Unicode characters
        filename = re.sub(r'[\uFEFF\u200B-\u200D\uFFFC]', '', filename)

        # Replace invalid characters with underscore
        invalid_chars = r'<>:"/\\|?*#\r\n\t'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Collapse multiple underscores
        filename = re.sub(r'_+', '_', filename)

        # Remove leading/trailing underscores and whitespace
        filename = filename.strip('_').strip()

        # Replace multiple dots with underscore
        filename = re.sub(r'\.{2,}', '_', filename)

        # Limit filename length
        return filename[:100] if filename else "无标题"

    async def _download_post_images(self, post: Dict, save_path: str) -> bool:
        """
        Download images for a post

        Args:
            post: Post dictionary
            save_path: Directory to save images

        Returns:
            True if any image was downloaded successfully
        """
        images_str = post.get('images', '')
        if not images_str:
            return False

        image_urls = [url.strip() for url in images_str.split(';') if url.strip()]
        if not image_urls:
            return False

        # Limit to 3 images
        image_urls = image_urls[:3]
        print(f"        发现 {len(image_urls)} 张图片，开始下载...")

        success_count = 0
        for i, url in enumerate(image_urls):
            try:
                success = await self._download_file_async(url, save_path, f"image_{i}.jpg")
                if success:
                    success_count += 1
            except Exception as e:
                print(f"        ✗ 图片 {i} 下载失败: {e}")

        if success_count > 0:
            print(f"        ✅ 成功下载 {success_count}/{len(image_urls)} 张图片")
            return True
        return False

    async def _download_post_videos(self, post: Dict, save_path: str) -> bool:
        """
        Download videos for a post

        Args:
            post: Post dictionary
            save_path: Directory to save videos

        Returns:
            True if any video was downloaded successfully
        """
        videos_str = post.get('videos', '')
        if not videos_str:
            return False

        video_urls = [url.strip() for url in videos_str.split(';') if url.strip()]
        if not video_urls:
            return False

        # Limit to 1 video
        video_urls = video_urls[:1]
        print(f"        发现 {len(video_urls)} 个视频，开始下载...")

        success_count = 0
        for i, url in enumerate(video_urls):
            try:
                success = await self._download_file_async(url, save_path, f"video_{i}.mp4")
                if success:
                    success_count += 1
            except Exception as e:
                print(f"        ✗ 视频 {i} 下载失败: {e}")

        if success_count > 0:
            print(f"        ✅ 成功下载 {success_count}/{len(video_urls)} 个视频")
            return True
        return False

    async def _download_file_async(self, url: str, save_path: str, filename: str) -> bool:
        """Download file asynchronously"""
        try:
            await asyncio.sleep(random.uniform(0.5, 1.0))  # Rate limiting

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://bbs.hupu.com/',
            }

            # Use requests in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(url, headers=headers, stream=True, timeout=30)
            )
            response.raise_for_status()

            # Create directory and save file
            os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, filename)

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Validate file size
            file_size = os.path.getsize(filepath)
            if file_size < 1000:  # Too small, likely an error
                os.remove(filepath)
                print(f"        ✗ 文件太小 ({file_size} bytes)，可能是错误")
                return False

            return True

        except Exception as e:
            print(f"        ✗ 文件下载失败: {e}")
            return False

    def get_all_download_users(self) -> List[str]:
        """
        Get all user IDs that exist in the download/hupu directory

        Returns:
            List of user IDs
        """
        hupu_download_dir = os.path.join(self.download_path, "hupu")

        if not os.path.exists(hupu_download_dir):
            print(f"虎扑下载目录不存在: {hupu_download_dir}")
            return []

        # Get all subdirectories (these are user IDs)
        user_dirs = []
        try:
            for item in os.listdir(hupu_download_dir):
                item_path = os.path.join(hupu_download_dir, item)
                if os.path.isdir(item_path):
                    user_dirs.append(item)

            print(f"发现 {len(user_dirs)} 个用户目录: {user_dirs}")
            return user_dirs

        except Exception as e:
            print(f"扫描下载目录失败: {e}")
            return []

    def process_all_users(self, save_data: bool = True, download_media: bool = True, max_users: int = None) -> Dict[str, List[Dict]]:
        """
        Process realtime data for all users

        Args:
            save_data: Whether to save the fetched data
            download_media: Whether to download the actual media files
            max_users: Maximum number of users to process (None for all)

        Returns:
            Dictionary mapping user_id to list of new posts
        """
        print("🚀 开始处理所有虎扑用户的实时数据...")
        print("=" * 60)

        # Get all users from dataset directory
        all_users = self.get_all_download_users()

        if not all_users:
            print("❌ 没有找到任何用户")
            return {}

        # Limit users if specified
        if max_users and len(all_users) > max_users:
            all_users = all_users[:max_users]
            print(f"限制处理用户数量到 {max_users} 个")

        print(f"准备处理 {len(all_users)} 个用户")

        results = {}
        total_new_posts = 0
        successful_users = 0
        failed_users = 0

        for i, user_id in enumerate(all_users, 1):
            print(f"\n📋 处理用户 {i}/{len(all_users)}: {user_id}")
            print("-" * 40)

            try:
                # Process this user's realtime data
                new_posts = self(user_id, save_data=save_data, download_media=download_media)
                results[user_id] = new_posts

                if new_posts:
                    total_new_posts += len(new_posts)
                    successful_users += 1
                    print(f"✅ 用户 {user_id} 处理完成: {len(new_posts)} 条新回帖")
                else:
                    print(f"📝 用户 {user_id} 没有新回帖")

                # Add delay between users to be polite
                if i < len(all_users):  # Don't delay after the last user
                    delay = random.uniform(3, 5)
                    print(f"⏳ 等待 {delay:.1f} 秒后处理下一个用户...")
                    time.sleep(delay)

            except Exception as e:
                print(f"❌ 处理用户 {user_id} 时出错: {e}")
                results[user_id] = []
                failed_users += 1
                continue

        print("\n" + "=" * 60)
        print("🎉 所有用户处理完成！")
        print(f"📊 总结:")
        print(f"  - 处理用户数: {len(all_users)}")
        print(f"  - 成功用户数: {successful_users}")
        print(f"  - 失败用户数: {failed_users}")
        print(f"  - 总新回帖数: {total_new_posts}")
        print(f"  - 有新内容的用户: {len([r for r in results.values() if r])}")
        print("=" * 60)

        return results

    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a user"""
        # Count existing posts
        user_dirs = [
            os.path.join(self.dataset_path, user_id),
            os.path.join(self.realtime_dataset_path, user_id)
        ]

        total_posts = 0
        for user_dir in user_dirs:
            if os.path.exists(user_dir):
                csv_files = [f for f in os.listdir(user_dir) if f.endswith('.csv')]
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(os.path.join(user_dir, csv_file))
                        total_posts += len(df)
                    except:
                        pass

        return {
            "user_id": user_id,
            "total_posts": total_posts,
            "dataset_paths": user_dirs
        }

