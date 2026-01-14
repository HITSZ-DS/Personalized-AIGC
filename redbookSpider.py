import requests
import re
import json
from json5 import loads as json5_loads
import csv
import os
import time
import random
import shutil
from typing import List, Dict, Optional, Set
from collections import deque
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class XiaohongshuCreatedNotesSpider:
    def __init__(self, cookies: str = None, max_users: int = 100):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.xiaohongshu.com/',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # 延迟设置（更保守）
        self.min_delay = 1.2
        self.max_delay = 1.5
        self.timeout = 20

        self.max_users = max_users
        # 移除原来的采样设置，使用脚本2的逻辑
        self.min_notes_threshold = 3  # 最少3篇才处理
        self.max_sample_size = 10     # 最大采样10篇

        self.processed_users: Set[str] = set()
        self.failed_users: Set[str] = set()
        self.user_queue = deque()
        self.success_count = 0
        self.failed_count = 0
        self.total_notes = 0

        # 线程安全计数器
        self.consecutive_failures = 0
        self.max_consecutive_failures = 12
        self.failure_lock = threading.Lock()
        self.last_success_time = time.time()
        self.recent_failed_users = deque(maxlen=12)

        self.export_format = "csv"
        self.enable_download = False

        # 进度文件
        self.progress_file = "xhs_created_notes_progress.json"

        if cookies:
            self._add_cookies(cookies)

    def _add_cookies(self, cookie_str: str):
        cookies = {}
        for item in cookie_str.split(';'):
            key_val = item.strip().split('=', 1)
            if len(key_val) == 2:
                cookies[key_val[0]] = key_val[1]
        self.session.cookies.update(cookies)

    def _random_delay(self):
        time.sleep(random.uniform(self.min_delay, self.max_delay))

    def _make_request(self, url: str) -> Optional[str]:
        self._random_delay()
        try:
            print(f"[请求] {url}")
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"[请求失败] {e}")
            return None

    def get_explore_page_users(self) -> List[str]:
        """从发现页获取用户ID"""
        url = 'https://www.xiaohongshu.com/explore'
        html = self._make_request(url)
        if not html:
            return []
        patterns = [
            r'"userInfo":\{"userId":"([a-f0-9]+)"',
            r'user/profile/([a-f0-9]+)',
            r'"userId":"([a-f0-9]{20,})"'
        ]
        found = set()
        for p in patterns:
            found.update(re.findall(p, html))
        return list(found)[:20]

    def get_user_created_notes(self, user_id: str) -> Optional[List[Dict]]:
        """获取用户创建的笔记"""
        url = f'https://www.xiaohongshu.com/user/profile/{user_id}'
        html = self._make_request(url)
        if not html:
            return None

        # 针对创建笔记
        m = re.search(r'"notes":\s*(\[\[.*?\]\])', html, re.DOTALL)
        if not m:
            return None

        try:
            notes_data = json5_loads(m.group(1))
            if len(notes_data) > 0 and notes_data[0]:  # 创建笔记在第一层
                all_notes = notes_data[0]

                # 采样逻辑
                sampled_notes = self._apply_note_sampling_logic(all_notes, user_id)
                if not sampled_notes:
                    return None

                formatted = []
                for n in sampled_notes:
                    # 处理笔记卡片数据
                    nc = n.get('noteCard', {}) if isinstance(n, dict) else n
                    if not isinstance(nc, dict):
                        continue

                    uid = nc.get('user', {}).get('userId', '') if isinstance(nc.get('user'), dict) else ''
                    nid = n.get('id', '') or nc.get('noteId', '')
                    token = n.get('xsecToken', '') or nc.get('xsecToken', '')

                    formatted.append({
                        'id': nid,
                        'title': nc.get('displayTitle', nc.get('title', '无标题')),
                        'type': nc.get('type', '未知类型'),
                        'user_nickname': nc.get('user', {}).get('nickname', '匿名用户'),
                        'user_id': uid or '未知ID',
                        'liked_count': self._safe_int(nc.get('interactInfo', {}).get('likedCount', '0')),
                        'cover_url': self._extract_cover_url(nc.get('cover', {})),
                        'note_url': f"https://www.xiaohongshu.com/explore/{nid}?xsec_token={token}" if nid else "",
                        'xsec_token': token,
                        'timestamp': int(time.time()),
                        'source_user_id': user_id
                    })
                return formatted
        except Exception as e:
            print(f"[解析错误] {e}")
        return None

    def _apply_note_sampling_logic(self, all_notes: List, user_id: str) -> Optional[List]:
        """笔记采样逻辑"""
        valid_notes = []
        for note in all_notes:
            # 过滤掉无效笔记
            if isinstance(note, dict) and (note.get('noteCard') or note.get('id')):
                valid_notes.append(note)

        print(f"[调试] 用户 {user_id} 的有效笔记数: {len(valid_notes)}")

        # 的采样规则
        if len(valid_notes) < self.min_notes_threshold:
            # 1. 小于3篇的，直接跳过
            print(f"[跳过] 用户 {user_id} 只有 {len(valid_notes)} 条笔记，少于{self.min_notes_threshold}条，跳过")
            return None
        elif len(valid_notes) <= 7:
            # 2. 3-7篇的，取全部
            sampled_notes = valid_notes
            print(f"[采样] 笔记数量 {len(valid_notes)} 在3-7之间，使用全部笔记")
        else:
            # 3. 大于7篇的，随机取7-10篇
            max_sample = min(self.max_sample_size, len(valid_notes))
            k = random.randint(7, max_sample)
            sampled_notes = random.sample(valid_notes, k)
            print(f"[采样] 笔记数量 {len(valid_notes)} 大于7，随机采样 {k} 条")

        return sampled_notes

    def get_note_detail(self, note_url: str) -> Optional[Dict]:
        """获取笔记详情（保留时间功能，但简化其他提取）"""
        # 延迟设置
        time.sleep(random.uniform(1.2, 1.5))

        html = self._make_request(note_url)
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        def get_meta(name, multi=False):
            tags = soup.find_all("meta", attrs={"name": name})
            if not tags: return [] if multi else None
            return [t["content"].strip() for t in tags if "content" in t.attrs] if multi else tags[0]["content"].strip()

        # 提取创建时间（保留核心功能）
        create_time = self._extract_create_time_simple(soup)

        return {
            "title": get_meta("og:title"),
            "description": get_meta("description"),
            "images": get_meta("og:image", multi=True),
            "note_type": get_meta("og:type"),
            "video_url": get_meta("og:video"),
            "video_time": get_meta("og:videotime"),
            "video_quality": get_meta("og:videoquality"),
            "like_count": get_meta("og:xhs:note_like"),
            "comment_count": get_meta("og:xhs:note_comment"),
            "collect_count": get_meta("og:xhs:note_collect"),
            "create_time": create_time,  # 保留创建时间
        }

    def _extract_create_time_simple(self, soup: BeautifulSoup) -> str:
        """简化版创建时间提取（避免复杂的DOM解析）"""
        try:
            # 方法1: 查找常见的日期元素
            date_selectors = ['.date', '.time', '.publish-time', '.note-time']
            for selector in date_selectors:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text().strip()
                    if text:
                        # 简单清理文本
                        clean_text = text.replace('编辑于', '').strip()
                        return self._normalize_time_format(clean_text)

            # 方法2: 在底部容器中查找
            bottom_containers = soup.find_all('div', class_='bottom-container')
            for container in bottom_containers:
                date_spans = container.find_all('span', class_='date')
                for span in date_spans:
                    text = span.get_text().strip()
                    if text:
                        clean_text = text.replace('编辑于', '').strip()
                        return self._normalize_time_format(clean_text)

        except Exception as e:
            print(f"[时间提取错误] {e}")

        return ""

    def _normalize_time_format(self, time_str: str) -> str:
        """标准化时间格式"""
        if not time_str:
            return ""

        base_time = time.time()

        # 处理包含具体时间的相对时间（如 "昨天 17:57"）
        if ' ' in time_str:
            parts = time_str.split(' ')
            if len(parts) >= 2:
                relative_part = parts[0]
                # 只处理相对时间部分，忽略具体时间
                return self._convert_single_relative_time(relative_part, base_time)

        return self._convert_single_relative_time(time_str, base_time)

    def _convert_single_relative_time(self, time_str: str, base_time: float) -> str:
        """转换单个相对时间字符串"""
        # 处理"昨天"
        if time_str == '昨天':
            yesterday = time.localtime(base_time - 24 * 60 * 60)
            return time.strftime("%Y-%m-%d", yesterday)

        # 处理"今天"
        if time_str == '今天':
            return time.strftime("%Y-%m-%d", time.localtime(base_time))

        # 处理"n天前"
        day_match = re.match(r'^(\d+)天前$', time_str)
        if day_match:
            days_ago = int(day_match.group(1))
            target_time = time.localtime(base_time - days_ago * 24 * 60 * 60)
            return time.strftime("%Y-%m-%d", target_time)

        # 处理"n小时前"
        hour_match = re.match(r'^(\d+)小时前$', time_str)
        if hour_match:
            hours_ago = int(hour_match.group(1))
            target_time = time.localtime(base_time - hours_ago * 60 * 60)
            return time.strftime("%Y-%m-%d", target_time)

        # 处理"n分钟前"
        minute_match = re.match(r'^(\d+)分钟前$', time_str)
        if minute_match:
            minutes_ago = int(minute_match.group(1))
            target_time = time.localtime(base_time - minutes_ago * 60)
            return time.strftime("%Y-%m-%d", target_time)

        # 处理"刚刚"
        if time_str == '刚刚':
            return time.strftime("%Y-%m-%d", time.localtime(base_time))

        return time_str

    def download_file(self, url: str, save_dir: str, filename: str):
        os.makedirs(save_dir, exist_ok=True)
        self._random_delay()
        try:
            print(f"[下载] {url}")
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            path = os.path.join(save_dir, filename)
            with open(path, "wb") as f:
                f.write(resp.content)
            return path
        except Exception as e:
            print(f"[下载失败] {url}, 错误: {e}")
            return None

    def enrich_with_details(self, notes: List[Dict], base_dir="downloads"):
        """丰富笔记详情"""
        enriched = []
        for note in notes:
            detail = self.get_note_detail(note["note_url"])
            if not detail:
                continue

            note_dir = os.path.join(base_dir, note["id"])
            local_imgs, local_video = [], None

            def normalize_url(url: str) -> Optional[str]:
                if not url:
                    return None
                if url.startswith("//picasso-static.xiaohongshu.com/fe-platform/"):
                    return None
                if url.startswith("//"):
                    return "https:" + url
                return url

            # 提取正文里的 #标签
            desc = detail.get("description") or ""
            tags = re.findall(r'#([^#\s]+)', desc)

            if self.enable_download:
                # 图文笔记：下载所有图片
                if detail["note_type"] != "video" and detail["images"]:
                    for idx, img in enumerate(detail["images"], 1):
                        img_url = normalize_url(img)
                        if not img_url:
                            continue
                        ext = os.path.splitext(img_url.split("?")[0])[1] or ".jpg"
                        p = self.download_file(img_url, note_dir, f"{note['id']}_img{idx}{ext}")
                        if p: local_imgs.append(p)

                # 视频笔记：下载视频 + 封面
                if detail["note_type"] == "video":
                    v_url = normalize_url(detail.get("video_url"))
                    if v_url:
                        local_video = self.download_file(v_url, note_dir, f"{note['id']}.mp4")
                    if detail["images"]:
                        cover = normalize_url(detail["images"][0])
                        if cover:
                            ext = os.path.splitext(cover.split("?")[0])[1] or ".jpg"
                            p = self.download_file(cover, note_dir, f"{note['id']}_cover{ext}")
                            if p: local_imgs.append(p)

            note.update({
                "description": desc,
                "tags": tags,
                "note_type": detail.get("note_type"),
                "images": detail.get("images") or [],
                "local_images": local_imgs,
                "video_url": detail.get("video_url"),
                "local_video": local_video,
                "video_time": detail.get("video_time"),
                "video_quality": detail.get("video_quality"),
                "like_count": detail.get("like_count"),
                "comment_count": detail.get("comment_count"),
                "collect_count": detail.get("collect_count"),
                "create_time": detail.get("create_time", ""),  # 保留创建时间
            })
            enriched.append(note)
        return enriched

    def _safe_int(self, v):
        try:
            if isinstance(v, str) and '万' in v: return int(float(v.replace('万',''))*10000)
            return int(v)
        except: return 0

    def _extract_cover_url(self, cover: Dict):
        if cover.get('urlDefault'): return cover['urlDefault']
        if cover.get('infoList'): return cover['infoList'][0].get('url','')
        return ''

    def export_results(self, data: List[Dict], user_id: str, format: str="csv"):
        timestamp = int(time.time())
        dir_name = f"xhs_created_notes/{user_id}_{timestamp}"
        os.makedirs(dir_name, exist_ok=True)

        if format == "csv":
            csv_path = os.path.join(dir_name, f"created_notes_{user_id}.csv")
            self._save_as_csv(data, csv_path)
            print(f"[导出] {csv_path}")
        else:
            json_path = os.path.join(dir_name, f"created_notes_{user_id}.json")
            with open(json_path, 'w', encoding='utf-8-sig') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[导出] {json_path}")

        # try:
        #     shutil.make_archive(dir_name, 'zip', dir_name)
        #     return f"{dir_name}.zip"
        # except:
        #     return dir_name
        return dir_name

    def _save_as_csv(self, notes, filepath):
        if not notes: return
        fields = [
            '序号','笔记ID','标题','类型','作者','作者ID',
            '点赞数','评论数','收藏数',
            '正文','话题标签',
            '封面URL','图片URL列表','本地图片路径列表',
            '视频URL','本地视频路径',
            '笔记URL','xsec_token','创建时间','采集时间'
        ]
        with open(filepath,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f); w.writerow(fields)
            for i,n in enumerate(notes,1):
                w.writerow([
                    i,n['id'],n['title'],
                    '视频' if n.get('note_type')=='video' else '图文',
                    n['user_nickname'],n['user_id'],
                    n.get('like_count'),n.get('comment_count'),n.get('collect_count'),
                    n.get('description',''),
                    "|".join(n.get('tags',[])),
                    n['cover_url'],
                    "|".join(n.get('images',[])),
                    "|".join(n.get('local_images',[])),
                    n.get('video_url',''),n.get('local_video',''),
                    n['note_url'],n.get('xsec_token',''),
                    n.get('create_time',''),
                    time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(n['timestamp']))
                ])

    def extract_author_ids_from_notes(self, notes: List[Dict]) -> List[str]:
        """从笔记中提取作者ID"""
        author_ids = set()
        for note in notes:
            author_id = note.get('user_id')
            if author_id and author_id != '未知ID' and author_id not in self.processed_users and author_id not in self.failed_users:
                author_ids.add(author_id)
        return list(author_ids)

    def _print_progress(self):
        """打印当前进度"""
        print(f"\n[进度] 成功: {self.success_count}/{self.max_users} | 失败: {self.failed_count} | 总笔记: {self.total_notes}")
        print(f"已处理: {len(self.processed_users)} | 待处理: {len(self.user_queue)}")
        print(f"连续失败: {self.consecutive_failures}/{self.max_consecutive_failures}")
        print("-" * 50)

    def save_progress(self):
        """保存进度到文件"""
        progress_data = {
            "processed_users": list(self.processed_users),
            "failed_users": list(self.failed_users),
            "user_queue": list(self.user_queue),
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "total_notes": self.total_notes,
            "max_users": self.max_users,
            "consecutive_failures": self.consecutive_failures,
            "recent_failed_users": list(self.recent_failed_users),
            "timestamp": time.time()
        }

        os.makedirs(os.path.dirname(self.progress_file) if os.path.dirname(self.progress_file) else ".", exist_ok=True)
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        print(f"[进度保存] 已保存到 {self.progress_file}")

    def load_progress(self):
        """从文件加载进度"""
        if not os.path.exists(self.progress_file):
            print("[进度加载] 无进度文件，从头开始")
            return False

        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)

            self.processed_users = set(progress_data.get("processed_users", []))
            self.failed_users = set(progress_data.get("failed_users", []))
            self.user_queue = deque(progress_data.get("user_queue", []))
            self.success_count = progress_data.get("success_count", 0)
            self.failed_count = progress_data.get("failed_count", 0)
            self.total_notes = progress_data.get("total_notes", 0)
            self.max_users = progress_data.get("max_users", self.max_users)
            self.consecutive_failures = progress_data.get("consecutive_failures", 0)
            self.recent_failed_users = deque(progress_data.get("recent_failed_users", []), maxlen=12)

            print(f"[进度加载] 已加载进度: {self.success_count} 成功用户, {self.failed_count} 失败用户, {len(self.user_queue)} 待处理用户")
            print(f"[进度加载] 连续失败: {self.consecutive_failures}")
            return True
        except Exception as e:
            print(f"[进度加载错误] {e}")
            return False

    def update_consecutive_failures(self, success: bool, user_id: str = None):
        """线程安全地更新连续失败计数器"""
        with self.failure_lock:
            if success:
                self.consecutive_failures = 0
                self.last_success_time = time.time()
                self.recent_failed_users.clear()
            else:
                self.consecutive_failures += 1
                if user_id:
                    self.recent_failed_users.append(user_id)

    def check_and_handle_blockage(self):
        """检查是否达到连续失败阈值并处理"""
        with self.failure_lock:
            if self.consecutive_failures >= self.max_consecutive_failures:
                print(f"⚠️ 警告: 所有线程连续失败 {self.consecutive_failures} 次，可能被网站拦截!")
                print(f"将最近 {len(self.recent_failed_users)} 个失败用户ID回退到待处理队列")

                for uid in reversed(self.recent_failed_users):
                    if (uid not in self.user_queue and
                        uid not in self.processed_users and
                        uid not in self.failed_users):
                        self.user_queue.appendleft(uid)
                        print(f"回退用户ID: {uid}")

                self.consecutive_failures = 0
                self.recent_failed_users.clear()
                self.save_progress()
                return True
        return False

    def _cleanup_failed_export(self, user_id: str):
        """清理失败的导出文件"""
        try:
            # 查找并删除该用户的所有导出文件
            export_pattern = f"xhs_created_notes/{user_id}_*"
            import glob
            failed_dirs = glob.glob(export_pattern)
            for dir_path in failed_dirs:
                if os.path.exists(dir_path):
                    if os.path.isdir(dir_path):
                        shutil.rmtree(dir_path)
                        print(f"[清理] 删除失败用户导出目录: {dir_path}")
                    # 同时删除对应的zip文件
                    zip_path = f"{dir_path}.zip"
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                        print(f"[清理] 删除失败用户导出zip: {zip_path}")
        except Exception as e:
            print(f"[清理失败] 无法删除用户 {user_id} 的导出文件: {e}")

    def _validate_enriched_notes(self, enriched_notes: List[Dict]) -> bool:
        """验证enriched notes是否包含关键信息"""
        if not enriched_notes:
            return False

        # 检查是否至少有一条笔记包含description和create_time
        valid_notes_count = 0
        for note in enriched_notes:
            description = note.get('description', '')
            create_time = note.get('create_time', '')
            # 如果description和create_time都不为空，则认为有效
            if description or create_time:
                valid_notes_count += 1

        # 如果有超过一半的笔记包含有效信息，则认为成功
        if valid_notes_count >= len(enriched_notes) * 0.5:
            return True
        return False

    def batch_crawl_parallel(self, max_workers=5):
        print("="*60)
        print("小红书用户创建笔记爬取 - 稳定版本")
        print("="*60)

        # 交互逻辑
        load_progress = input("是否加载之前的进度? (y/n, 默认y): ").strip().lower() != 'n'
        if load_progress:
            self.load_progress()

        max_users_input = input(f"最大用户数? (默认{self.max_users}): ").strip()
        if max_users_input:
            try:
                self.max_users = int(max_users_input)
            except ValueError:
                print(f"输入无效，保持默认值 {self.max_users}")

        export_choice = input("导出格式? 1=JSON 2=CSV (默认2): ").strip()
        self.export_format = "json" if export_choice=="1" else "csv"
        download_choice = input("是否下载图片/视频资源? (y/n, 默认n): ").strip().lower()
        self.enable_download = (download_choice == "y")

        # 初始化用户队列
        if not self.user_queue:
            explore_users = self.get_explore_page_users()
            for user_id in explore_users:
                if (user_id not in self.processed_users and
                    user_id not in self.failed_users and
                    user_id not in self.user_queue):
                    self.user_queue.append(user_id)
            print(f"从发现页获取了 {len(explore_users)} 个用户，过滤后添加 {len(self.user_queue)} 个新用户")

        # 采用脚本1的并行处理逻辑，但添加线程启动间隔和脚本2的验证逻辑
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while self.success_count < self.max_users and self.user_queue:
                batch_users = [self.user_queue.popleft() for _ in range(min(max_workers, len(self.user_queue)))]

                # 使用轮转机制启动线程，每个线程间隔2秒
                futures = {}
                for i, uid in enumerate(batch_users):
                    # 每个线程间隔2秒启动（只在启动时加间隔）
                    if i > 0:
                        time.sleep(2)
                    future = executor.submit(self.crawl_single_user, uid)
                    futures[future] = uid
                    print(f"[线程启动] 线程 {i+1}/{len(batch_users)} 已启动，处理用户 {uid}")

                for future in as_completed(futures):
                    uid = futures[future]
                    try:
                        result = future.result()
                        if result:
                            uid, enriched, notes = result

                            # 关键验证：检查enriched notes是否包含有效信息
                            if not self._validate_enriched_notes(enriched):
                                print(f"[失败] 用户 {uid} 的笔记缺少关键信息(description和create_time)，视为失败")
                                self._cleanup_failed_export(uid)  # 清理导出文件
                                self.failed_users.add(uid)
                                self.failed_count += 1
                                self.update_consecutive_failures(False, uid)
                                continue

                            # 导出结果
                            self.export_results(enriched, uid, self.export_format)

                            self.success_count += 1
                            self.total_notes += len(enriched)
                            self.processed_users.add(uid)
                            self.update_consecutive_failures(True)

                            # 提取作者ID加入队列
                            author_ids = self.extract_author_ids_from_notes(notes)
                            for author_id in author_ids:
                                if (author_id not in self.processed_users and
                                    author_id not in self.failed_users and
                                    author_id not in self.user_queue):
                                    self.user_queue.append(author_id)
                                    print(f"添加新用户到队列: {author_id}")

                            print(f"✓ 成功处理用户 {uid}: {len(enriched)} 篇创建笔记")
                        else:
                            print(f"用户 {uid} 笔记获取失败")
                            self.failed_users.add(uid)
                            self.failed_count += 1
                            self.update_consecutive_failures(False, uid)
                    except Exception as e:
                        print(f"用户 {uid} 出错: {e}")
                        self.failed_users.add(uid)
                        self.failed_count += 1
                        self.update_consecutive_failures(False, uid)

                # 检查是否达到连续失败阈值
                if self.check_and_handle_blockage():
                    input("程序已暂停。请检查是否被网站拦截，按回车继续或Ctrl+C退出...")
                    continue

                # 打印进度并保存
                self._print_progress()
                self.save_progress()

                # 如果队列空了但还没达到最大用户数，尝试获取更多用户
                if not self.user_queue and self.success_count < self.max_users:
                    print("队列为空，尝试从发现页获取更多用户...")
                    explore_users = self.get_explore_page_users()
                    added_count = 0
                    for user_id in explore_users:
                        if (user_id not in self.processed_users and
                            user_id not in self.failed_users and
                            user_id not in self.user_queue):
                            self.user_queue.append(user_id)
                            added_count += 1
                    print(f"从发现页获取了 {len(explore_users)} 个用户，过滤后添加 {added_count} 个新用户")

        print(f"\n任务完成! 成功: {self.success_count}, 失败: {self.failed_count}, 总创建笔记: {self.total_notes}")

    def crawl_single_user(self, uid):
        """单用户爬取逻辑"""
        time.sleep(random.uniform(self.min_delay, self.max_delay))

        notes = self.get_user_created_notes(uid)
        if not notes:
            return None

        enriched = self.enrich_with_details(notes, base_dir="downloads")
        if not enriched:
            return None
        return (uid, enriched, notes)

def main():
    cookies = input("请输入你的小红书cookies (直接回车跳过): ").strip()
    spider = XiaohongshuCreatedNotesSpider(cookies if cookies else None)
    spider.batch_crawl_parallel(max_workers=2)

if __name__ == "__main__":
    main()
