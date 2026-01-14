#!/usr/bin/env python3
"""
小红书实时爬取脚本 V2
采用 HTML + Selenium 策略，确保笔记 URL 包含 xsec_token，防止被封
支持下载图片、视频及文本内容
"""

import os
import time
import json
import random
import re
import asyncio
import aiohttp
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from json5 import loads as json5_loads
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

try:
    import dateutil.parser
except ImportError:
    print("警告: dateutil未安装，时间解析功能可能受限")
    dateutil = None


# ===================== 配置区域 =====================

# 路径配置
DATASET_PATH = "dataset/redbook"
REALTIME_DATASET_PATH = "dataset/redbook_realtime"
DOWNLOAD_PATH = "download"
PROGRESS_FILE = "download/redbook/realtime_progress.json"  # 断点续传进度文件

# Cookie（需要手动更新）
COOKIE_STR = "abRequestId=a9bac428-fbf4-5b70-ae9d-ae6e82bdf264; a1=19ab03e0dd71s5v6v33vw0myyf60nk2m7pu6v2t3d50000423372; webId=137269e03e381ab4a60bb862e4a8e3a8; gid=yj0D8qdJYdW8yj0D8qd8fUJ9fWyM2hKhqqhE7qf86vviKK288ITJ6h8884JqqWJ82fYyf8Sy; acw_tc=0a00d93117683708775673137eeb31c0233e6cfbe594960a47b13671d228c9; webBuild=5.7.0; websectiga=82e85efc5500b609ac1166aaf086ff8aa4261153a448ef0be5b17417e4512f28; sec_poison_id=60138e5a-9a57-45d5-92f2-67f52cb2a7ed; web_session=040069b77f21a94aaeedce70523b4b48831700; id_token=VjEAANeK14HCDRX4yJALANNegYAiBmeb5Bssr0GYghCCG34SH+W6r4dURI+Iqec8ANlCzWJTHFkzc2eHfHXwEkjRJJWJLkkpnwz8C7KEW+Yc2ehZ5NE6mvWtFzzPtzqcQF9j/t5G; xsecappid=xhs-pc-web; unread={%22ub%22:%226960d8d9000000000a02b5ef%22%2C%22ue%22:%22696461a2000000002103c341%22%2C%22uc%22:34}; loadts=1768372432540"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.xiaohongshu.com/",
    "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
}

# 下载配置
MAX_IMAGES_PER_NOTE = 5  # 每个笔记最多下载图片数
MAX_NOTES_PER_USER = 3   # 每个用户最多获取新笔记数
DOWNLOAD_TIMEOUT = 30
REQUEST_DELAY = (2, 4)   # 请求间隔（秒）

# Selenium 配置
USE_SELENIUM = True
HEADLESS = True
MAX_SELENIUM_PAGES = 4   # 作者主页最多翻页数
SELENIUM_TIMEOUT = 60    # Selenium 总超时时间（秒）

# =====================================================


def sleep_random():
    """随机延迟"""
    time.sleep(random.uniform(*REQUEST_DELAY))


def normalize_title(title: str) -> str:
    """标准化标题：去空格、转小写"""
    if not title:
        return ""
    return re.sub(r"\s+", "", title).lower().strip()


class RedBookRealtime:
    def __init__(self, cookies: str = None):
        """
        Initialize the RedBookRealtime crawler
        
        Args:
            cookies: Optional cookies for authenticated requests
        """
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        
        # Use provided cookies or default
        cookie_str = cookies or COOKIE_STR
        if cookie_str:
            self._add_cookies(cookie_str)
        
        self.cookies = cookie_str
        
        # Paths configuration
        self.dataset_path = DATASET_PATH
        self.realtime_dataset_path = REALTIME_DATASET_PATH
        self.download_path = DOWNLOAD_PATH
        self.progress_file = PROGRESS_FILE
        
        # Statistics
        self.stats = {
            'users_processed': 0,
            'new_notes_found': 0,
            'notes_downloaded': 0,
            'images_downloaded': 0,
            'videos_downloaded': 0,
            'failed_downloads': 0,
            'skipped_notes': 0  # 跳过的已下载笔记数
        }
        
        # 断点续传：加载进度（新格式：按用户组织）
        self.progress = self._load_progress()  # {user_id: {completed, notes: {...}}}
        self.progress_changed = False
        
        # 重定向计数器：用于智能切换到Selenium模式
        self.redirect_count = 0
        self.use_selenium_fallback = False
    
    def _add_cookies(self, cookie_str):
        """添加 Cookies"""
        for item in cookie_str.split(';'):
            if '=' in item:
                k, v = item.strip().split('=', 1)
                self.session.cookies.set(k, v)
    
    def _load_progress(self) -> dict:
        """加载断点续传进度（新格式：按用户组织）"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 新格式：{user_id: {completed, notes: {note_id: {...}}}}
                    if isinstance(data, dict):
                        # 检查是否是新格式（包含用户ID作为key）
                        is_new_format = False
                        for key, value in data.items():
                            if isinstance(value, dict) and 'notes' in value:
                                is_new_format = True
                                break
                        
                        if is_new_format:
                            # 新格式，直接使用
                            total_users = len(data)
                            completed_users = sum(1 for u in data.values() if u.get('completed', False))
                            total_notes = sum(len(u.get('notes', {})) for u in data.values())
                            print(f"✓ 已加载断点续传进度: {total_users} 个用户，{total_notes} 个笔记，{completed_users} 个用户已完成")
                            return data
                        else:
                            # 旧格式（{note_id: {user_id, ...}}），转换为新格式
                            print(f"⚠️  检测到旧格式进度文件，正在转换...")
                            new_data = {}
                            for note_id, info in data.items():
                                user_id = info.get('user_id', 'unknown')
                                if user_id not in new_data:
                                    new_data[user_id] = {
                                        'completed': False,
                                        'last_update': info.get('download_time', ''),
                                        'timestamp': info.get('timestamp', 0),
                                        'notes': {}
                                    }
                                new_data[user_id]['notes'][note_id] = {
                                    'status': info.get('status', 'success'),
                                    'timestamp': info.get('timestamp', 0),
                                    'download_time': info.get('download_time', '')
                                }
                            print(f"✓ 已转换: {len(new_data)} 个用户")
                            return new_data
                    else:
                        print(f"⚠️  无法识别的进度文件格式，将从头开始")
                        return {}
            except Exception as e:
                print(f"⚠️  加载进度文件失败: {e}，将从头开始")
                return {}
        else:
            print(f"ℹ️  进度文件不存在，将创建新的进度记录")
            return {}
    
    def _save_progress(self):
        """保存断点续传进度"""
        if not self.progress_changed:
            return
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
            
            # 保存详细信息（新格式）
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
            
            self.progress_changed = False
            print(f"    ✓ 进度已保存到: {self.progress_file}")
        except Exception as e:
            print(f"    ⚠️  保存进度失败: {e}")
    
    def _is_user_completed(self, user_id: str) -> bool:
        """检查用户是否已完成（找到所有最新笔记）"""
        if user_id not in self.progress:
            return False
        return self.progress[user_id].get('completed', False)
    
    def _is_note_downloaded(self, user_id: str, note_id: str) -> bool:
        """检查指定用户的笔记是否已下载"""
        if user_id not in self.progress:
            return False
        notes = self.progress[user_id].get('notes', {})
        return note_id in notes
    
    def _mark_note_downloaded(self, note_id: str, user_id: str, status: str = "success"):
        """标记笔记为已下载"""
        # 确保用户存在
        if user_id not in self.progress:
            self.progress[user_id] = {
                'completed': False,
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': int(time.time()),
                'notes': {}
            }
        
        # 添加笔记
        if note_id not in self.progress[user_id]['notes']:
            self.progress[user_id]['notes'][note_id] = {
                'status': status,
                'timestamp': int(time.time()),
                'download_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.progress[user_id]['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S')
            self.progress[user_id]['timestamp'] = int(time.time())
            self.progress_changed = True
    
    def _mark_user_completed(self, user_id: str, completed: bool = True):
        """标记用户为已完成（已找到所有最新笔记）"""
        if user_id not in self.progress:
            self.progress[user_id] = {
                'completed': completed,
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': int(time.time()),
                'notes': {}
            }
        else:
            self.progress[user_id]['completed'] = completed
            self.progress[user_id]['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S')
            self.progress[user_id]['timestamp'] = int(time.time())
        
        self.progress_changed = True
    
    def clear_progress(self, user_id: str = None):
        """清空进度文件（用于重新开始）
        
        Args:
            user_id: 如果指定，只清空该用户的进度；否则清空所有进度
        """
        try:
            if user_id:
                # 只清空指定用户
                if user_id in self.progress:
                    del self.progress[user_id]
                    self.progress_changed = True
                    self._save_progress()
                    print(f"✓ 已清空用户 {user_id} 的进度")
            else:
                # 清空所有
                if os.path.exists(self.progress_file):
                    os.remove(self.progress_file)
                    print(f"✓ 已清空进度文件: {self.progress_file}")
                self.progress = {}
                self.progress_changed = False
        except Exception as e:
            print(f"❌ 清空进度失败: {e}")
    
    def get_progress_stats(self) -> dict:
        """获取进度统计信息"""
        if not hasattr(self, 'progress') or not self.progress:
            return {'total_users': 0, 'total_notes': 0}
        
        stats = {
            'total_users': len(self.progress),
            'completed_users': 0,
            'incomplete_users': 0,
            'total_notes': 0,
            'by_user': {},
            'by_status': {'success': 0, 'failed': 0}
        }
        
        for user_id, user_data in self.progress.items():
            is_completed = user_data.get('completed', False)
            notes = user_data.get('notes', {})
            note_count = len(notes)
            
            # 统计完成状态
            if is_completed:
                stats['completed_users'] += 1
            else:
                stats['incomplete_users'] += 1
            
            # 统计笔记数
            stats['total_notes'] += note_count
            stats['by_user'][user_id] = {
                'note_count': note_count,
                'completed': is_completed,
                'last_update': user_data.get('last_update', '')
            }
            
            # 统计状态
            for note_id, note_info in notes.items():
                status = note_info.get('status', 'success')
                if status in stats['by_status']:
                    stats['by_status'][status] += 1
        
        return stats
    
    def __call__(self, user_id: str, save_data: bool = True, download_media: bool = True) -> List[Dict]:
        """
        Get latest notes for a specific user (同步版本)
        
        Args:
            user_id: The real xiaohongshu user ID
            save_data: Whether to automatically save the fetched data
            download_media: Whether to download the actual media files
        
        Returns:
            List of latest notes (max 3)
        """
        print(f"\n{'='*60}")
        print(f"处理用户: {user_id}")
        print(f"{'='*60}")
        
        # Get latest timestamp from existing data
        latest_timestamp = self._get_latest_timestamp(user_id)
        
        # Fetch latest notes with HTML + Selenium
        new_notes = self._fetch_latest_notes(user_id, latest_timestamp)
        
        if not new_notes:
            print(f"用户 {user_id} 没有找到新笔记")
            return []
        
        print(f"\n✓ 找到 {len(new_notes)} 个新笔记")
        
        # Automatically save data if requested
        if save_data and new_notes:
            self._save_realtime_data(new_notes, user_id)
        
        # Download media files if requested
        if download_media and new_notes:
            print(f"\n开始下载用户 {user_id} 的实时媒体文件...")
            asyncio.run(self._download_realtime_media(new_notes, user_id))
        
        self.stats['users_processed'] += 1
        self.stats['new_notes_found'] += len(new_notes)
        
        # 标记用户为已完成（已找到所有最新笔记）
        self._mark_user_completed(user_id, completed=True)
        self._save_progress()
        print(f"\n✓ 用户 {user_id} 已标记为完成")
        
        # 重置重定向计数器，为下一个用户准备
        self.redirect_count = 0
        self.use_selenium_fallback = False
        
        return new_notes
    
    async def process_user_async(self, user_id: str, save_data: bool = True, download_media: bool = True) -> List[Dict]:
        """
        Get latest notes for a specific user (异步版本，用于并行处理)
        
        Args:
            user_id: The real xiaohongshu user ID
            save_data: Whether to automatically save the fetched data
            download_media: Whether to download the actual media files
        
        Returns:
            List of latest notes (max 3)
        """
        print(f"\n{'='*60}")
        print(f"[异步] 处理用户: {user_id}")
        print(f"{'='*60}")
        
        # Get latest timestamp from existing data (同步操作，在线程池中执行)
        loop = asyncio.get_event_loop()
        latest_timestamp = await loop.run_in_executor(None, self._get_latest_timestamp, user_id)
        
        # Fetch latest notes with HTML + Selenium (同步操作，在线程池中执行)
        new_notes = await loop.run_in_executor(None, self._fetch_latest_notes, user_id, latest_timestamp)
        
        if not new_notes:
            print(f"[异步] 用户 {user_id} 没有找到新笔记")
            return []
        
        print(f"\n[异步] ✓ 用户 {user_id} 找到 {len(new_notes)} 个新笔记")
        
        # Automatically save data if requested
        if save_data and new_notes:
            await loop.run_in_executor(None, self._save_realtime_data, new_notes, user_id)
        
        # Download media files if requested
        if download_media and new_notes:
            print(f"\n[异步] 开始下载用户 {user_id} 的实时媒体文件...")
            await self._download_realtime_media(new_notes, user_id)
        
        self.stats['users_processed'] += 1
        self.stats['new_notes_found'] += len(new_notes)
        
        # 标记用户为已完成（已找到所有最新笔记）
        await loop.run_in_executor(None, self._mark_user_completed, user_id, True)
        await loop.run_in_executor(None, self._save_progress)
        print(f"\n[异步] ✓ 用户 {user_id} 已标记为完成")
        
        # 重置重定向计数器，为下一个用户准备
        self.redirect_count = 0
        self.use_selenium_fallback = False
        
        return new_notes
    
    def _get_latest_timestamp(self, user_id: str) -> int:
        """Get the latest fav_time timestamp from user's existing data"""
        print(f"\n→ 检查用户 {user_id} 的最新时间戳...")
        
        user_dirs = []
        
        # Check main dataset path
        main_user_dir = os.path.join(self.dataset_path, user_id)
        if os.path.exists(main_user_dir):
            user_dirs.append(main_user_dir)
        
        # Check realtime dataset path
        realtime_user_dir = os.path.join(self.realtime_dataset_path, user_id)
        if os.path.exists(realtime_user_dir):
            user_dirs.append(realtime_user_dir)
        
        if not user_dirs:
            print(f"  用户数据目录不存在，将获取所有笔记")
            return 0
        
        latest_timestamp = 0
        try:
            for user_dir in user_dirs:
                # Find all CSV files
                csv_files = [f for f in os.listdir(user_dir) if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    csv_path = os.path.join(user_dir, csv_file)
                    try:
                        df = pd.read_csv(csv_path)
                        if 'fav_time' in df.columns and not df.empty:
                            for fav_time in df['fav_time']:
                                if pd.notna(fav_time):
                                    timestamp = self._convert_to_timestamp(fav_time)
                                    if timestamp:
                                        latest_timestamp = max(latest_timestamp, timestamp)
                    except Exception as e:
                        print(f"  读取CSV失败 {csv_path}: {e}")
                        continue
        
        except Exception as e:
            print(f"  扫描目录失败: {e}")
        
        print(f"  最新时间戳: {latest_timestamp}")
        if latest_timestamp > 0:
            time_str = datetime.fromtimestamp(latest_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  对应时间: {time_str}")
        
        return latest_timestamp
    
    def _convert_to_timestamp(self, time_value) -> int:
        """Convert various time formats to timestamp"""
        if isinstance(time_value, (int, float)):
            return int(time_value)
        
        if isinstance(time_value, str):
            try:
                if time_value.isdigit():
                    return int(time_value)
                
                if dateutil:
                    dt = dateutil.parser.parse(time_value)
                    return int(dt.timestamp())
            except:
                return 0
        
        return 0
    
    # ========== HTML + Selenium 爬取逻辑 ==========
    
    def _get_author_profile_url(self, user_id):
        """获取作者主页 URL（不需要 token）"""
        return f"https://www.xiaohongshu.com/user/profile/{user_id}"
    
    def parse_notes_from_json(self, json_data):
        """从 JSON 数据解析笔记信息"""
        notes = []
        if not json_data:
            return notes
        
        data_list = []
        if isinstance(json_data, list):
            if len(json_data) > 0 and isinstance(json_data[0], list):
                data_list = json_data[0]
            else:
                data_list = json_data
        elif isinstance(json_data, dict):
            if "notes" in json_data:
                return self.parse_notes_from_json(json_data["notes"])
            if "data" in json_data and "notes" in json_data["data"]:
                return self.parse_notes_from_json(json_data["data"]["notes"])
        
        for item in data_list:
            if not isinstance(item, dict):
                continue
            
            note_card = item.get("noteCard", {})
            note_id = item.get("id") or note_card.get("noteId")
            title = (note_card.get("displayTitle") or 
                    note_card.get("title") or 
                    item.get("displayTitle") or "")
            
            # 提取笔记类型
            note_type = note_card.get("type", "")
            
            # 提取互动数据
            interact_info = note_card.get("interactInfo", {})
            liked_count = interact_info.get("likedCount", 0)
            collected_count = interact_info.get("collectedCount", 0)
            comment_count = interact_info.get("commentCount", 0)
            
            # 提取封面
            cover_url = ""
            if "cover" in note_card:
                cover = note_card["cover"]
                if isinstance(cover, dict) and "urlDefault" in cover:
                    cover_url = cover["urlDefault"]
                elif isinstance(cover, str):
                    cover_url = cover
            
            # 提取 URL（包含 xsec_token）
            url = None
            
            # 方法1: 检查现成的 URL 字段
            if isinstance(note_card, dict):
                for key in note_card.keys():
                    if 'url' in key.lower() or 'link' in key.lower() or 'href' in key.lower():
                        url_value = note_card[key]
                        if url_value and isinstance(url_value, str):
                            if 'xsec_token' in url_value or '/explore/' in url_value or '/user/profile/' in url_value:
                                url = url_value
                                break
            
            if not url:
                for key in item.keys():
                    if 'url' in key.lower() or 'link' in key.lower() or 'href' in key.lower():
                        url_value = item[key]
                        if url_value and isinstance(url_value, str):
                            if 'xsec_token' in url_value or '/explore/' in url_value or '/user/profile/' in url_value:
                                url = url_value
                                break
            
            # 方法2: 使用 xsecToken 构建完整 URL
            if not url and note_id:
                xsec_token = None
                if isinstance(note_card, dict) and 'xsecToken' in note_card:
                    xsec_token = note_card['xsecToken']
                elif 'xsecToken' in item:
                    xsec_token = item['xsecToken']
                
                if xsec_token:
                    url = f"https://www.xiaohongshu.com/explore/{note_id}?xsec_token={xsec_token}"
            
            if note_id:
                notes.append({
                    "id": note_id,
                    "title": title,
                    "title_norm": normalize_title(title),
                    "url": url,
                    "type": note_type,
                    "cover_url": cover_url,
                    "liked_count": liked_count,
                    "collected_count": collected_count,
                    "comment_count": comment_count
                })
        
        return notes
    
    def fetch_notes_from_html(self, user_id):
        """方法1: HTML 请求获取笔记列表"""
        url = self._get_author_profile_url(user_id)
        print(f"  → HTML 请求: {url}")
        
        sleep_random()
        
        try:
            resp = self.session.get(url, timeout=20)
            
            if resp.status_code != 200:
                print(f"     ⚠️  状态码: {resp.status_code}")
                return []
            
            # 提取 JSON
            match = re.search(r'"notes":\s*(\[\[.*?\]\])', resp.text, re.DOTALL)
            if match:
                try:
                    raw_json = json5_loads(match.group(1))
                    notes = self.parse_notes_from_json(raw_json)
                    print(f"     ✓ HTML 解析成功，找到 {len(notes)} 个笔记")
                    return notes
                except Exception as e:
                    print(f"     ⚠️  JSON 解析失败: {e}")
            else:
                print(f"     ⚠️  未找到 notes JSON 数据")
            
            return []
        
        except Exception as e:
            print(f"     ❌ HTML 请求失败: {e}")
            return []
    
    def _create_driver(self):
        """创建 Selenium WebDriver"""
        options = Options()
        if HEADLESS:
            options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument(f'user-agent={HEADERS["User-Agent"]}')
        
        driver = webdriver.Chrome(options=options)
        driver.set_window_size(1920, 1080)
        driver.implicitly_wait(5)
        driver.set_page_load_timeout(30)
        
        return driver
    
    def _add_cookies_to_driver(self, driver):
        """向 WebDriver 添加 Cookies"""
        driver.get("https://www.xiaohongshu.com/404")
        time.sleep(1)
        
        for item in self.cookies.split(';'):
            if '=' in item:
                name, value = item.strip().split('=', 1)
                try:
                    driver.add_cookie({
                        'name': name,
                        'value': value,
                        'domain': '.xiaohongshu.com'
                    })
                except:
                    pass
    
    def _extract_notes_from_page(self, driver):
        """从 Selenium 页面提取笔记"""
        notes = []
        
        # 尝试 JSON
        try:
            html = driver.page_source
            match = re.search(r'"notes":\s*(\[\[.*?\]\])', html, re.DOTALL)
            if match:
                raw = json5_loads(match.group(1))
                notes.extend(self.parse_notes_from_json(raw))
        except:
            pass
        
        # DOM 解析
        try:
            items = driver.find_elements(By.CSS_SELECTOR, "section.note-item")
            
            if len(items) == 0:
                alt_selectors = [".note-item", "a.cover", "[class*='note']", "[class*='Note']"]
                is_direct_link = False
                for selector in alt_selectors:
                    alt_items = driver.find_elements(By.CSS_SELECTOR, selector)
                    if alt_items:
                        items = alt_items
                        if selector == "a.cover":
                            is_direct_link = True
                        break
            else:
                is_direct_link = False
            
            for item in items:
                try:
                    if is_direct_link:
                        link_elem = item
                        href = item.get_attribute("href")
                    else:
                        link_elem = item.find_element(By.CSS_SELECTOR, "a.cover")
                        href = link_elem.get_attribute("href")
                    
                    if not href:
                        continue
                    
                    # 提取 note_id
                    note_id = ""
                    match_profile = re.search(r'/user/profile/[^/]+/([a-z0-9]+)', href)
                    match_explore = re.search(r'/explore/([a-z0-9]+)', href)
                    
                    if match_profile:
                        note_id = match_profile.group(1)
                    elif match_explore:
                        note_id = match_explore.group(1)
                    
                    if not note_id:
                        continue
                    
                    # 提取标题
                    title = ""
                    if not is_direct_link:
                        try:
                            title_elem = item.find_element(By.CSS_SELECTOR, ".footer .title span")
                            title = title_elem.text.strip()
                        except:
                            title = link_elem.get_attribute("title") or ""
                    else:
                        title = link_elem.get_attribute("title") or ""
                    
                    # 完整URL（包含xsec_token）
                    full_url = href if href.startswith("http") else "https://www.xiaohongshu.com" + href
                    
                    # 转换URL格式：/user/profile/{user_id}/{note_id}?params → /explore/{note_id}?params
                    if "/user/profile/" in full_url:
                        params = ""
                        if "?" in full_url:
                            params = "?" + full_url.split("?", 1)[1]
                        full_url = f"https://www.xiaohongshu.com/explore/{note_id}{params}"
                    
                    # 去重
                    existing_note = None
                    for n in notes:
                        if n['id'] == note_id:
                            existing_note = n
                            break
                    
                    if existing_note:
                        if not existing_note.get('url') and full_url:
                            existing_note['url'] = full_url
                            if title:
                                existing_note['title'] = title
                                existing_note['title_norm'] = normalize_title(title)
                    else:
                        notes.append({
                            "id": note_id,
                            "title": title,
                            "title_norm": normalize_title(title),
                            "url": full_url,
                            "type": "",
                            "cover_url": "",
                            "liked_count": 0,
                            "collected_count": 0,
                            "comment_count": 0
                        })
                
                except Exception as e:
                    continue
        
        except Exception as e:
            pass
        
        return notes
    
    def fetch_notes_with_selenium(self, user_id, max_notes: int = MAX_NOTES_PER_USER * 3):
        """方法2: Selenium 获取笔记列表"""
        start_time = time.time()
        
        print(f"  → Selenium 访问作者主页...")
        print(f"     目标: 获取最近 {max_notes} 个笔记")
        
        driver = None
        
        try:
            driver = self._create_driver()
            
            if time.time() - start_time > SELENIUM_TIMEOUT:
                print(f"     ⚠️  启动超时")
                return []
            
            self._add_cookies_to_driver(driver)
            
            url = self._get_author_profile_url(user_id)
            print(f"     访问: {url}")
            
            try:
                driver.get(url)
            except Exception as e:
                print(f"     ⚠️  页面加载超时: {e}")
                return []
            
            time.sleep(5)
            
            # 检查页面
            current_url = driver.current_url
            if '/login' in current_url or '/404' in current_url:
                print(f"     ⚠️  页面被重定向: {current_url}")
                return []
            
            all_notes = {}
            no_new_count = 0
            
            for page in range(MAX_SELENIUM_PAGES):
                elapsed = time.time() - start_time
                if elapsed > SELENIUM_TIMEOUT:
                    print(f"     ⚠️  总超时 ({elapsed:.1f}秒)，停止翻页")
                    break
                
                print(f"     翻页 {page + 1}/{MAX_SELENIUM_PAGES}...", end=" ", flush=True)
                
                try:
                    current_notes = self._extract_notes_from_page(driver)
                except Exception as e:
                    print(f"提取失败: {e}")
                    current_notes = []
                
                new_count = 0
                for note in current_notes:
                    if note['id'] not in all_notes:
                        all_notes[note['id']] = note
                        new_count += 1
                
                print(f"新增 {new_count} 个")
                
                # 检查是否足够
                if len(all_notes) >= max_notes:
                    print(f"     ✓ 已获取足够笔记 ({len(all_notes)} 个)")
                    break
                
                # 连续无新笔记，退出
                if new_count == 0:
                    no_new_count += 1
                    if no_new_count >= 2:
                        print(f"     ℹ️  连续 {no_new_count} 页无新笔记，停止翻页")
                        break
                else:
                    no_new_count = 0
                
                # 滚动加载
                if page < MAX_SELENIUM_PAGES - 1:
                    try:
                        last_height = driver.execute_script("return document.body.scrollHeight")
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(3)
                        
                        new_height = driver.execute_script("return document.body.scrollHeight")
                        if new_height == last_height:
                            print(f"     已到底部")
                            break
                    except Exception as e:
                        print(f"     滚动失败: {e}")
                        break
            
            print(f"     ✓ Selenium 完成，共获取 {len(all_notes)} 个笔记")
            
            return list(all_notes.values())
        
        except Exception as e:
            print(f"     ❌ Selenium 失败: {e}")
            return []
        
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def _fetch_latest_notes(self, user_id: str, latest_timestamp: int) -> List[Dict]:
        """
        Fetch latest notes that are newer than latest_timestamp
        使用 HTML + Selenium 策略，确保 URL 包含 xsec_token
        """
        print(f"\n→ 获取用户 {user_id} 的最新笔记...")
        
        all_notes = []
        
        # 阶段1: HTML 解析
        print(f"\n[阶段1] HTML 解析...")
        html_notes = self.fetch_notes_from_html(user_id)
        
        if html_notes:
            print(f"  ✓ HTML 获取到 {len(html_notes)} 个笔记")
            
            # 检查有多少笔记有完整 URL
            notes_with_url = sum(1 for n in html_notes if n.get('url'))
            print(f"  其中 {notes_with_url} 个笔记有完整 URL（含 token）")
            
            if notes_with_url < len(html_notes):
                print(f"  ⚠️  {len(html_notes) - notes_with_url} 个笔记缺少 URL，使用 Selenium 获取")
                selenium_notes = self.fetch_notes_with_selenium(user_id)
                
                if selenium_notes:
                    # 用 Selenium 的笔记补充 HTML 的笔记
                    selenium_by_id = {n['id']: n for n in selenium_notes if n.get('url')}
                    
                    for i, note in enumerate(html_notes):
                        if not note.get('url') and note['id'] in selenium_by_id:
                            html_notes[i] = selenium_by_id[note['id']]
                    
                    # 添加 Selenium 中新发现的笔记
                    existing_ids = {n['id'] for n in html_notes}
                    for note in selenium_notes:
                        if note['id'] not in existing_ids and note.get('url'):
                            html_notes.append(note)
            
            all_notes = html_notes
        else:
            # HTML 失败，直接用 Selenium
            print(f"  ⚠️  HTML 解析失败，使用 Selenium")
            selenium_notes = self.fetch_notes_with_selenium(user_id)
            
            if selenium_notes:
                print(f"  ✓ Selenium 获取到 {len(selenium_notes)} 个笔记")
                all_notes = selenium_notes
            else:
                print(f"  ❌ Selenium 也失败")
                return []
        
        if not all_notes:
            return []
        
        # 验证 URL
        notes_with_valid_url = []
        for note in all_notes:
            if note.get('url'):
                notes_with_valid_url.append(note)
            else:
                # 如果没有 URL，尝试构建基础 URL
                note['url'] = f"https://www.xiaohongshu.com/explore/{note['id']}"
                print(f"  ⚠️  笔记 {note['id']} 缺少 token，使用基础 URL")
                notes_with_valid_url.append(note)
        
        print(f"\n→ 获取笔记详细信息...")
        
        # 为所有笔记获取详细信息（包括创建时间）
        enriched_notes = []
        selenium_driver = None  # Selenium driver，仅在需要时创建
        
        for i, note in enumerate(notes_with_valid_url, 1):
            print(f"  [{i}/{len(notes_with_valid_url)}] {note['title'][:30]}...")
            
            # 智能选择：如果HTML模式连续失败4次，切换到Selenium模式
            if self.use_selenium_fallback:
                if selenium_driver is None:
                    print(f"    🚀 初始化Selenium浏览器...")
                    selenium_driver = self._init_driver()
                
                note_detail = self._get_note_detail_with_selenium(selenium_driver, note)
            else:
                # 默认使用HTML模式
                note_detail = self._get_note_detail(note)
            
            if note_detail:
                enriched_notes.append(note_detail)
            
            # 限制数量（获取比需要的多一些，因为有些可能不是新的）
            if len(enriched_notes) >= MAX_NOTES_PER_USER * 2:
                break
            
            # 延迟
            if i < len(notes_with_valid_url):
                time.sleep(random.uniform(1, 2))
        
        # 清理Selenium driver
        if selenium_driver:
            try:
                selenium_driver.quit()
                print(f"    ✅ 已关闭Selenium浏览器")
            except:
                pass
        
        # 筛选新笔记
        new_notes = []
        for note in enriched_notes:
            note_timestamp = note.get('fav_time', 0)
            if note_timestamp > latest_timestamp:
                new_notes.append(note)
                if len(new_notes) >= MAX_NOTES_PER_USER:
                    break
        
        # 按时间戳排序（最新的在前）
        new_notes.sort(key=lambda x: x.get('fav_time', 0), reverse=True)
        
        return new_notes
    
    def _get_note_detail(self, note: Dict) -> Optional[Dict]:
        """获取笔记详细信息（HTML模式）"""
        note_url = note.get('url', '')
        note_id = note.get('id', '')
        
        if not note_url:
            return None
        
        sleep_random()
        
        try:
            resp = self.session.get(note_url, timeout=20)
            
            # 检查重定向
            if '/404' in resp.url or '/login' in resp.url:
                self.redirect_count += 1
                print(f"    ⚠️  被重定向 (连续第{self.redirect_count}次): {resp.url[:80]}...")
                
                # 当连续重定向超过4次时，触发切换到Selenium模式
                if self.redirect_count >= 4 and not self.use_selenium_fallback:
                    self.use_selenium_fallback = True
                    print(f"    🔄 连续重定向{self.redirect_count}次，切换到Selenium模式...")
                
                return None
            
            if resp.status_code != 200:
                print(f"    ⚠️  状态码: {resp.status_code}")
                return None
            
            # 成功获取，重置重定向计数器
            if self.redirect_count > 0:
                print(f"    ✅ 成功获取详情，重置重定向计数器")
                self.redirect_count = 0
                self.use_selenium_fallback = False
            
            # 解析详情
            detail = self._parse_note_detail(resp.text, note)
            return detail
        
        except Exception as e:
            print(f"    ❌ 获取详情失败: {e}")
            return None
    
    def _get_note_detail_with_selenium(self, driver, note: Dict) -> Optional[Dict]:
        """使用Selenium获取笔记详细信息"""
        note_url = note.get('url', '')
        note_id = note.get('id', '')
        
        if not note_url:
            return None
        
        try:
            print(f"    🌐 使用Selenium访问: {note_url[:60]}...")
            driver.get(note_url)
            time.sleep(random.uniform(3, 5))
            
            # 检查是否被重定向
            current_url = driver.current_url
            if '/login' in current_url or '/404' in current_url:
                print(f"    ⚠️  Selenium也被重定向: {current_url[:80]}...")
                return None
            
            # 获取页面HTML
            html = driver.page_source
            
            # 解析详情
            detail = self._parse_note_detail(html, note)
            
            # 成功获取，重置计数器
            if detail:
                self.redirect_count = 0
                print(f"    ✅ Selenium成功获取详情")
            
            return detail
        
        except Exception as e:
            print(f"    ❌ Selenium获取详情失败: {e}")
            return None
    
    def _parse_note_detail(self, html: str, base_note: Dict) -> Dict:
        """解析笔记详情页"""
        soup = BeautifulSoup(html, "html.parser")
        
        # 提取创建时间
        create_time = ""
        timestamp = int(time.time())
        
        # 尝试从多个可能的位置提取时间
        time_patterns = [
            r'"time":\s*"([^"]+)"',
            r'"createTime":\s*"([^"]+)"',
            r'"publishTime":\s*"([^"]+)"',
            r'"updateTime":\s*(\d+)',
            r'发布于\s*(\d{4}-\d{2}-\d{2})',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, html)
            if match:
                time_str = match.group(1)
                try:
                    if time_str.isdigit():
                        timestamp = int(time_str)
                        if timestamp > 10000000000:  # 毫秒转秒
                            timestamp = timestamp // 1000
                    elif dateutil:
                        dt = dateutil.parser.parse(time_str)
                        timestamp = int(dt.timestamp())
                    create_time = time_str
                    break
                except:
                    continue
        
        # 提取描述/内容
        description = ""
        
        # 方法1: meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            description = meta_desc.get("content", "")
        
        # 方法2: 从 JSON 中提取
        if not description:
            desc_match = re.search(r'"desc":\s*"([^"]+)"', html)
            if desc_match:
                description = desc_match.group(1)
        
        # 提取图片URLs
        image_urls = self._extract_images_from_note_page(html)
        
        # 提取视频URL（尝试多个可能的字段）
        video_url = ""
        video_patterns = [
            r'"videoUrl":\s*"([^"]+)"',
            r'"video":\s*{\s*"[^"]*url[^"]*":\s*"([^"]+)"',
            r'"streamUrl":\s*"([^"]+)"',
            r'"playUrl":\s*"([^"]+)"',
            r'"originVideoKey":\s*"([^"]+)"'
        ]
        
        for pattern in video_patterns:
            video_match = re.search(pattern, html)
            if video_match:
                video_url = video_match.group(1)
                # 验证是否是有效的视频 URL
                if video_url and ('video' in video_url.lower() or 'stream' in video_url.lower()):
                    break
        
        # 如果还没找到，尝试从 JSON 数据中递归查找
        if not video_url:
            video_url = self._find_video_in_html(html)
        
        # 如果还没找到，尝试从HTML中直接搜索完整的视频URL
        if not video_url:
            # 搜索常见的视频URL格式
            video_url_patterns = [
                r'(https?://[^"\s]*sns-video[^"\s]*\.mp4)',
                r'(https?://[^"\s]*stream[^"\s]*\.mp4)',
                r'(https?://[^"\s]*video[^"\s]*\.mp4)',
            ]
            for pattern in video_url_patterns:
                video_match = re.search(pattern, html)
                if video_match:
                    video_url = video_match.group(1)
                    break
        
        # 构建完整笔记信息
        return {
            'redbookID': base_note.get('id', ''),
            'title': base_note.get('title', ''),
            'note_url': base_note.get('url', ''),
            'content': description,
            'fav_time': timestamp,
            'user_id': '',  # 从URL或其他地方提取
            'type': base_note.get('type', ''),
            'user_nickname': '',
            'liked_count': base_note.get('liked_count', 0),
            'comment_count': base_note.get('comment_count', 0),
            'collect_count': base_note.get('collected_count', 0),
            'tags': '',
            'cover_url': base_note.get('cover_url', ''),
            'images': '|'.join(image_urls),
            'video_url': video_url,
            'xsec_token': self._extract_xsec_token(base_note.get('url', '')),
            'create_time': create_time,
            'timestamp': int(time.time())
        }
    
    def _extract_xsec_token(self, url: str) -> str:
        """从 URL 中提取 xsec_token"""
        if not url:
            return ""
        
        match = re.search(r'xsec_token=([^&]+)', url)
        if match:
            return match.group(1)
        
        return ""
    
    def _extract_images_from_note_page(self, html: str) -> List[str]:
        """从笔记详情页提取图片URL（改进版）"""
        soup = BeautifulSoup(html, "html.parser")
        image_urls = []
        
        # 方法1: 从 script 标签中的 JSON 提取
        script_images = self._extract_images_from_script_tags(html)
        image_urls.extend(script_images)
        
        # 方法2: og:image meta 标签
        for meta in soup.find_all("meta"):
            name_attr = meta.get("name")
            prop_attr = meta.get("property")
            if name_attr == "og:image" or prop_attr == "og:image":
                content = meta.get("content")
                if content and self._is_valid_note_image(content):
                    if content not in image_urls:
                        image_urls.append(content)
        
        # 方法3: 从整个 HTML 中正则搜索
        try:
            image_pattern = r'"(https?://[^"]*xhscdn\.com[^"]*)"'
            matches = re.findall(image_pattern, html)
            for url in matches:
                if self._is_valid_note_image(url) and url not in image_urls:
                    image_urls.append(url)
        except:
            pass
        
        # 处理URL并去重，最后再过滤一次确保没有视频
        unique_urls = []
        seen = set()
        for url in image_urls:
            if url.startswith("//"):
                url = "https:" + url
            # 最终过滤：排除视频URL
            if any(keyword in url.lower() for keyword in ['video', 'stream', '.mp4', '.m3u8']):
                continue
            if url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        return unique_urls
    
    def _extract_images_from_script_tags(self, html: str) -> List[str]:
        """从 JavaScript 数据中提取图片"""
        images = []
        
        patterns = [
            r'"imageList":\s*(\[.*?\])',
            r'"images":\s*(\[.*?\])',
            r'window\.__INITIAL_STATE__\s*=\s*({.*?})</script>'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    found_images = self._find_images_in_json_data(data)
                    images.extend(found_images)
                except:
                    continue
        
        return images
    
    def _find_images_in_json_data(self, data) -> List[str]:
        """递归查找 JSON 数据中的图片 URL"""
        images = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() in ['url', 'image', 'cover', 'pic', 'urldefault', 'traceurl'] and isinstance(value, str):
                    if value.startswith('http') and 'xhscdn.com' in value and self._is_valid_note_image(value):
                        images.append(value)
                elif isinstance(value, (dict, list)):
                    images.extend(self._find_images_in_json_data(value))
        elif isinstance(data, list):
            for item in data:
                images.extend(self._find_images_in_json_data(item))
        
        return images
    
    def _find_video_in_html(self, html: str) -> str:
        """从 HTML 中查找视频 URL"""
        # 尝试从 JSON 数据中递归查找
        try:
            # 查找可能包含视频数据的 JSON
            json_pattern = r'window\.__INITIAL_STATE__\s*=\s*({.*?})</script>'
            match = re.search(json_pattern, html, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                video_url = self._find_video_in_json_data(data)
                if video_url:
                    return video_url
        except:
            pass
        
        return ""
    
    def _find_video_in_json_data(self, data) -> str:
        """递归查找 JSON 数据中的视频 URL"""
        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() in ['videourl', 'streamurl', 'playurl', 'url', 'video', 'mp4url', 'h264url'] and isinstance(value, str):
                    # 验证是视频URL
                    if value.startswith('http') and (
                        'video' in value.lower() or 
                        'stream' in value.lower() or 
                        '.mp4' in value.lower() or
                        '.m3u8' in value.lower()
                    ):
                        return value
                elif isinstance(value, (dict, list)):
                    result = self._find_video_in_json_data(value)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = self._find_video_in_json_data(item)
                if result:
                    return result
        
        return ""
    
    def _is_valid_note_image(self, url: str) -> bool:
        """验证是否为有效的笔记图片"""
        if not url or url.startswith("data:image"):
            return False
        
        # 排除视频URL
        if any(keyword in url.lower() for keyword in ['video', 'stream', '.mp4', '.m3u8']):
            return False
        
        # 排除JS/CSS文件
        if any(ext in url.lower() for ext in ['.js', '.css', '.json']):
            return False
        
        # 排除纯域名（没有路径的URL）
        if url.count('/') <= 3:  # https://domain.com/ 只有3个斜杠
            return False
        
        blacklist = ["logo", "icon", "avatar", "favicon", "default", "placeholder"]
        if any(b in url.lower() for b in blacklist):
            return False
        
        if any(p in url for p in ["/fe-platform/", "/fe-static/", "/static/", "/as/v1/", "/formula-static/"]):
            return False
        
        if "xhscdn.com" not in url:
            return False
        
        # 确保是图片相关的域名或路径
        valid_patterns = ['webpic', 'image', '.jpg', '.png', '.webp', '.jpeg', 'nd_dft', 'nd_prv', 'nc_n']
        if not any(pattern in url.lower() for pattern in valid_patterns):
            return False
        
        return True
    
    # ========== 数据保存 ==========
    
    def _save_realtime_data(self, notes: List[Dict], user_id: str):
        """保存实时数据为 CSV"""
        if not notes:
            return
        
        timestamp = int(time.time())
        user_dir = os.path.join(self.realtime_dataset_path, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        csv_filename = f"realtime_notes_{timestamp}.csv"
        csv_filepath = os.path.join(user_dir, csv_filename)
        
        self._save_as_csv(notes, csv_filepath)
        
        print(f"\n✓ 数据已保存: {csv_filepath}")
        print(f"  笔记数: {len(notes)}")
    
    def _save_as_csv(self, notes: List[Dict], filepath: str):
        """保存为 CSV 文件"""
        if not notes:
            return
        
        import csv
        
        fieldnames = [
            '序号', 'redbookID', 'title', '作者', '作者ID',
            '点赞数', '评论数', '收藏数', 'content', 'tag', '封面URL',
            'images', '本地图片路径列表', 'videos', '本地视频路径',
            '笔记URL', 'xsec_token', 'fav_time', '采集时间'
        ]
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, note in enumerate(notes, 1):
                csv_row = {
                    '序号': i,
                    'redbookID': note.get('redbookID', ''),
                    'title': note.get('title', ''),
                    '作者': note.get('user_nickname', ''),
                    '作者ID': note.get('user_id', ''),
                    '点赞数': note.get('liked_count', 0),
                    '评论数': note.get('comment_count', 0),
                    '收藏数': note.get('collect_count', 0),
                    'content': note.get('content', ''),
                    'tag': note.get('tags', ''),
                    '封面URL': note.get('cover_url', ''),
                    'images': note.get('images', ''),
                    '本地图片路径列表': '',
                    'videos': note.get('video_url', ''),
                    '本地视频路径': '',
                    '笔记URL': note.get('note_url', ''),
                    'xsec_token': note.get('xsec_token', ''),
                    'fav_time': note.get('fav_time', ''),
                    '采集时间': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(note.get('timestamp', time.time())))
                }
                writer.writerow(csv_row)
    
    # ========== 媒体下载 ==========
    
    async def _download_realtime_media(self, notes: List[Dict], user_id: str):
        """下载实时媒体文件（带断点续传）"""
        if not notes:
            return
        
        realtime_dir = os.path.join(self.download_path, "redbook", user_id, "realtime")
        os.makedirs(realtime_dir, exist_ok=True)
        
        print(f"\n下载目录: {realtime_dir}")
        
        for i, note in enumerate(notes, 1):
            note_id = note.get('redbookID', f'note_{i}')
            note_title = note.get('title', '无标题')
            
            # 断点续传：检查是否已下载
            if self._is_note_downloaded(user_id, note_id):
                print(f"\n[{i}/{len(notes)}] ⏭️  跳过已下载笔记: {note_title[:50]}... (ID: {note_id})")
                self.stats['skipped_notes'] += 1
                continue
            
            print(f"\n[{i}/{len(notes)}] 处理笔记: {note_title[:50]}...")
            
            note_folder = os.path.join(realtime_dir, note_id)
            os.makedirs(note_folder, exist_ok=True)
            
            try:
                # 保存文本
                self._save_text_content(note_title, note.get('content', ''), note_folder)
                
                # 下载媒体
                await self._download_note_media(note, note_folder)
                
                # 标记为已下载
                self._mark_note_downloaded(note_id, user_id, status="success")
                self._save_progress()
                
                self.stats['notes_downloaded'] += 1
                print(f"  ✅ 笔记处理完成（已保存进度）")
            
            except Exception as e:
                print(f"  ❌ 笔记处理失败: {e}")
                # 标记为失败（但仍然记录，避免重复尝试）
                self._mark_note_downloaded(note_id, user_id, status="failed")
                self._save_progress()
                self.stats['failed_downloads'] += 1
            
            if i < len(notes):
                await asyncio.sleep(random.uniform(1, 2))
    
    def _save_text_content(self, title: str, content: str, save_path: str) -> bool:
        """保存文本内容"""
        try:
            content_clean = re.sub(r'#\w+', '', content)
            content_clean = re.sub(r'\s+', ' ', content_clean).strip()
            
            if not content_clean:
                content_clean = 'No content'
            
            clean_title = self._sanitize_filename(title)
            filename = f"{clean_title}.txt"
            filepath = os.path.join(save_path, filename)
            
            if len(content_clean) > 2000:
                content_clean = content_clean[:2000]
            
            full_content = f"Title: {title}\n\nContent: {content_clean}"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            print(f"    ✅ 文本保存成功")
            return True
        
        except Exception as e:
            print(f"    ⚠️  文本保存失败: {e}")
            return False
    
    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
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
    
    async def _download_note_media(self, note: Dict, note_folder: str):
        """下载笔记的媒体文件（图片和视频）"""
        video_url = note.get('video_url', '')
        
        # 下载图片
        image_urls_str = note.get('images', '')
        if image_urls_str:
            image_urls = image_urls_str.split('|')
            image_urls = [url for url in image_urls if url]
            
            if image_urls:
                # 如果有视频，只下载第一张封面图（高清）
                if video_url:
                    print(f"    下载视频封面（1张）...")
                    await self._download_images([image_urls[0]], note_folder)
                else:
                    # 没有视频，下载所有图片
                    print(f"    下载 {len(image_urls)} 张图片...")
                    await self._download_images(image_urls, note_folder)
        
        # 下载视频
        if video_url:
            print(f"    下载视频...")
            await self._download_video(video_url, note_folder)
    
    async def _download_images(self, image_urls: List[str], save_path: str):
        """异步下载图片（带并发控制）"""
        # 限制并发数，避免连接过多
        semaphore = asyncio.Semaphore(3)  # 最多同时下载3张
        
        async def download_with_limit(session, url, filepath, index):
            async with semaphore:
                return await self._download_file_async(session, url, filepath, "image")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i, img_url in enumerate(image_urls[:MAX_IMAGES_PER_NOTE]):
                ext = self._get_file_extension(img_url)
                filename = f"image_{i}{ext}"
                filepath = os.path.join(save_path, filename)
                
                task = download_with_limit(session, img_url, filepath, i)
                tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success_count = sum(1 for r in results if r is True)
                print(f"      ✓ 成功下载 {success_count}/{len(tasks)} 张图片")
                self.stats['images_downloaded'] += success_count
    
    async def _download_video(self, video_url: str, save_path: str):
        """异步下载视频"""
        async with aiohttp.ClientSession() as session:
            filename = "video_0.mp4"
            filepath = os.path.join(save_path, filename)
            
            success = await self._download_file_async(session, video_url, filepath, "video")
            if success:
                print(f"      ✓ 视频下载成功")
                self.stats['videos_downloaded'] += 1
            else:
                print(f"      ✗ 视频下载失败")
    
    async def _download_file_async(self, session: aiohttp.ClientSession, url: str, save_path: str, file_type: str = "file", max_retries: int = 3) -> bool:
        """异步下载文件（带重试机制）"""
        if os.path.exists(save_path):
            # 检查已存在文件的大小
            file_size = os.path.getsize(save_path)
            if file_size >= 1000:  # 如果文件有效，跳过
                return False
            else:  # 如果文件太小，删除并重新下载
                os.remove(save_path)
        
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(random.uniform(0.5, 1.0))
                
                headers = {
                    'User-Agent': HEADERS['User-Agent'],
                    'Referer': 'https://www.xiaohongshu.com/',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8' if file_type == "image" else '*/*',
                }
                
                async with session.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        
                        # 验证文件大小
                        content_size = len(content)
                        if content_size < 1000:
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)  # 指数退避
                                continue
                            return False
                        
                        # 保存文件
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        with open(save_path, 'wb') as f:
                            f.write(content)
                        
                        # 验证文件完整性（图片）
                        if file_type == "image":
                            if not self._verify_image(save_path):
                                os.remove(save_path)
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(2 ** attempt)
                                    continue
                                return False
                        
                        return True
                    else:
                        # 非200状态码，重试
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
            
            except asyncio.TimeoutError:
                # 超时，重试
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
            
            except Exception as e:
                # 其他异常，重试
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
        
        return False
    
    def _verify_image(self, filepath: str) -> bool:
        """验证图片文件完整性"""
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                img.verify()
            return True
        except:
            # 如果 PIL 不可用或验证失败，使用简单的文件头检查
            try:
                with open(filepath, 'rb') as f:
                    header = f.read(12)
                    # 检查常见图片格式的魔数
                    if header[:3] == b'\xff\xd8\xff':  # JPEG
                        return True
                    elif header[:8] == b'\x89PNG\r\n\x1a\n':  # PNG
                        return True
                    elif header[:6] in (b'GIF87a', b'GIF89a'):  # GIF
                        return True
                    elif header[:4] == b'RIFF' and header[8:12] == b'WEBP':  # WEBP
                        return True
                return False
            except:
                return False
    
    def _get_file_extension(self, url: str) -> str:
        """获取文件扩展名"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path
        
        if '.' in path:
            ext = '.' + path.split('.')[-1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                return ext
        
        return '.jpg'
    
    # ========== 批量处理 ==========
    
    def get_all_download_users(self) -> List[str]:
        """获取所有下载目录中的用户ID"""
        redbook_download_dir = os.path.join(self.download_path, "redbook")
        
        if not os.path.exists(redbook_download_dir):
            return []
        
        user_dirs = []
        try:
            for item in os.listdir(redbook_download_dir):
                item_path = os.path.join(redbook_download_dir, item)
                if os.path.isdir(item_path):
                    user_dirs.append(item)
            
            return user_dirs
        
        except Exception as e:
            print(f"扫描下载目录失败: {e}")
            return []
    
    def process_all_users(self, save_data: bool = True, download_media: bool = True, max_users: int = None, parallel: int = 3) -> Dict[str, List[Dict]]:
        """
        处理所有用户的实时数据
        
        Args:
            save_data: 是否保存数据
            download_media: 是否下载媒体文件
            max_users: 最多处理用户数（限制爬取的用户数量）
            parallel: 并发数量（默认3，同时处理3个用户）
        """
        if parallel > 1:
            # 使用异步并行处理
            return asyncio.run(self._process_all_users_async(save_data, download_media, max_users, parallel))
        else:
            # 使用原有的串行处理
            return self._process_all_users_sync(save_data, download_media, max_users)
    
    def _process_all_users_sync(self, save_data: bool = True, download_media: bool = True, max_users: int = None) -> Dict[str, List[Dict]]:
        """串行处理所有用户（原逻辑）"""
        print("🚀 开始批量处理用户实时数据（串行模式）...")
        print("="*60)
        
        download_users = self.get_all_download_users()
        
        if not download_users:
            print("❌ 没有找到任何用户")
            return {}
        
        if max_users and len(download_users) > max_users:
            download_users = download_users[:max_users]
        
        print(f"准备处理 {len(download_users)} 个用户\n")
        
        results = {}
        
        for i, user_id in enumerate(download_users, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(download_users)}] 处理用户: {user_id}")
            print(f"{'='*60}")
            
            # 检查用户是否已完成（断点续传）
            if self._is_user_completed(user_id):
                print(f"⏭️  用户 {user_id} 已完成，跳过")
                results[user_id] = []
                continue
            
            try:
                new_notes = self(user_id, save_data=save_data, download_media=download_media)
                results[user_id] = new_notes
                
                if new_notes:
                    print(f"\n✅ 用户 {user_id} 完成: {len(new_notes)} 个新笔记")
                else:
                    print(f"\n📝 用户 {user_id} 没有新笔记")
                
                # 延迟
                if i < len(download_users):
                    delay = random.uniform(4, 5)
                    print(f"\n⏳ 等待 {delay:.1f} 秒...")
                    time.sleep(delay)
            
            except Exception as e:
                print(f"\n❌ 处理用户 {user_id} 失败: {e}")
                results[user_id] = []
                continue
        
        # 显示统计
        self._show_stats()
        
        return results
    
    async def _process_all_users_async(self, save_data: bool = True, download_media: bool = True, max_users: int = None, parallel: int = 2) -> Dict[str, List[Dict]]:
        """并行处理所有用户"""
        print(f"🚀 开始批量处理用户实时数据（并行模式：{parallel} 个并发）...")
        print("="*60)
        
        download_users = self.get_all_download_users()
        
        if not download_users:
            print("❌ 没有找到任何用户")
            return {}
        
        if max_users and len(download_users) > max_users:
            download_users = download_users[:max_users]
        
        print(f"准备处理 {len(download_users)} 个用户\n")
        
        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(parallel)
        results = {}
        
        async def process_with_limit(user_id: str, index: int, total: int):
            """带并发限制的用户处理"""
            async with semaphore:
                print(f"\n{'='*60}")
                print(f"[{index}/{total}] 🔄 开始处理用户: {user_id}")
                print(f"{'='*60}")
                
                # 检查用户是否已完成（断点续传）
                if self._is_user_completed(user_id):
                    print(f"⏭️  用户 {user_id} 已完成，跳过")
                    return []
                
                try:
                    new_notes = await self.process_user_async(user_id, save_data=save_data, download_media=download_media)
                    
                    if new_notes:
                        print(f"\n✅ [{index}/{total}] 用户 {user_id} 完成: {len(new_notes)} 个新笔记")
                    else:
                        print(f"\n📝 [{index}/{total}] 用户 {user_id} 没有新笔记")
                    
                    return user_id, new_notes
                
                except Exception as e:
                    print(f"\n❌ [{index}/{total}] 处理用户 {user_id} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return user_id, []
        
        # 创建所有任务
        tasks = [
            process_with_limit(user_id, i+1, len(download_users))
            for i, user_id in enumerate(download_users)
        ]
        
        # 执行所有任务
        print(f"\n🚀 开始并行处理 {len(tasks)} 个用户（并发数: {parallel}）...\n")
        
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整理结果
        for result in completed_results:
            if isinstance(result, Exception):
                print(f"⚠️  任务异常: {result}")
                continue
            
            user_id, notes = result
            results[user_id] = notes
        
        # 显示统计
        self._show_stats()
        
        return results
    
    def _show_stats(self):
        """显示统计信息"""
        print("\n" + "="*60)
        print("📊 处理统计")
        print("="*60)
        print(f"处理用户数:      {self.stats['users_processed']}")
        print(f"发现新笔记数:    {self.stats['new_notes_found']}")
        print(f"下载笔记数:      {self.stats['notes_downloaded']}")
        print(f"跳过笔记数:      {self.stats['skipped_notes']} (已下载)")
        print(f"失败笔记数:      {self.stats['failed_downloads']}")
        print(f"下载图片数:      {self.stats['images_downloaded']}")
        print(f"下载视频数:      {self.stats['videos_downloaded']}")
        print("="*60)
        
        # 显示进度统计
        if hasattr(self, 'progress') and len(self.progress) > 0:
            progress_stats = self.get_progress_stats()
            print(f"📁 断点续传统计")
            print(f"  总用户数:     {progress_stats['total_users']}")
            print(f"  已完成用户:   {progress_stats['completed_users']}")
            print(f"  未完成用户:   {progress_stats['incomplete_users']}")
            print(f"  总已下载笔记: {progress_stats['total_notes']}")
            print(f"  成功:         {progress_stats['by_status'].get('success', 0)}")
            print(f"  失败:         {progress_stats['by_status'].get('failed', 0)}")
            print(f"  进度文件:     {self.progress_file}")
            print("="*60)
        
        print("✅ 完成！")


# =====================================================

if __name__ == "__main__":
    import sys
    
    crawler = RedBookRealtime()
    
    # 检查是否需要清空进度
    if '--clear-progress' in sys.argv:
        print("🗑️  清空断点续传进度...")
        crawler.clear_progress()
        print("✓ 进度已清空，将重新下载所有内容")
        sys.exit(0)
    
    # 检查是否需要查看进度
    if '--show-progress' in sys.argv:
        print("📊 断点续传进度统计:")
        stats = crawler.get_progress_stats()
        print(f"  总用户数:     {stats['total_users']}")
        print(f"  已完成用户:   {stats['completed_users']}")
        print(f"  未完成用户:   {stats['incomplete_users']}")
        print(f"  总已下载笔记: {stats['total_notes']}")
        print(f"  成功:         {stats['by_status'].get('success', 0)}")
        print(f"  失败:         {stats['by_status'].get('failed', 0)}")
        print(f"\n按用户统计:")
        for user_id, user_info in stats['by_user'].items():
            status_icon = "✅" if user_info['completed'] else "🔄"
            print(f"    {status_icon} {user_id}: {user_info['note_count']} 个笔记 (最后更新: {user_info['last_update']})")
        print(f"\n进度文件: {crawler.progress_file}")
        sys.exit(0)
    
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        # 处理指定用户
        user_id = sys.argv[1]
        crawler(user_id, save_data=True, download_media=True)
    else:
        # 批量处理所有用户
        # 使用方法：
        # python tools/redbook_realtime.py                            # 默认3个并行
        # python tools/redbook_realtime.py --parallel 2               # 2个并行
        # python tools/redbook_realtime.py --parallel 5               # 5个并行
        # python tools/redbook_realtime.py --max-users 200            # 只处理前200个用户
        # python tools/redbook_realtime.py --parallel 3 --max-users 200  # 3个并行，最多200个用户
        # python tools/redbook_realtime.py --clear-progress           # 清空进度，重新开始
        # python tools/redbook_realtime.py --show-progress            # 查看进度统计
        
        parallel = 3  # 默认3个并行
        max_users = None  # 默认处理所有用户
        
        # 检查是否指定了并发数
        if '--parallel' in sys.argv:
            try:
                idx = sys.argv.index('--parallel')
                if idx + 1 < len(sys.argv):
                    parallel = int(sys.argv[idx + 1])
                    print(f"✓ 设置并发数: {parallel}")
            except (ValueError, IndexError):
                print("⚠️  并发数参数错误，使用默认值 3")
        
        # 检查是否指定了用户上限
        if '--max-users' in sys.argv:
            try:
                idx = sys.argv.index('--max-users')
                if idx + 1 < len(sys.argv):
                    max_users = int(sys.argv[idx + 1])
                    print(f"✓ 设置用户上限: {max_users}")
            except (ValueError, IndexError):
                print("⚠️  用户上限参数错误，将处理所有用户")
        
        crawler.process_all_users(save_data=True, download_media=True, max_users=max_users, parallel=parallel)

