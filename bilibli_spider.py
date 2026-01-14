import requests
import json
import time
import os
import random
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
import re
import multiprocessing
from functools import partial

class BilibiliFavoritesSpider:
    def __init__(self, cookies: str = None):
        # 基本请求头配置
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.bilibili.com/',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # 添加Cookie支持
        if cookies:
            self._add_cookies(cookies)

    def _add_cookies(self, cookie_str: str):
        """解析并添加Cookie"""
        cookies = {}
        for item in cookie_str.split(';'):
            key_val = item.strip().split('=', 1)
            if len(key_val) == 2:
                cookies[key_val[0]] = key_val[1]

        self.cookies = cookies
        self.session.cookies.update(cookies)

    def make_api_request(self, url: str, params: dict = None) -> Optional[Dict]:
        """API请求方法"""
        try:
            # 显示基本信息用于调试
            print(f"\n[请求] URL: {url}")
            if params:
                print(f"[参数] {params}")

            response = self.session.get(url, params=params, timeout=20)
            print(f"[状态] {response.status_code}")

            # 限制日志输出长度
            preview = response.text[:300] + "..." if len(response.text) > 300 else response.text
            print(f"[响应] {preview}")

            response.raise_for_status()

            return response.json() if 'application/json' in response.headers.get('Content-Type', '') else None

        except requests.exceptions.RequestException as e:
            print(f"[错误] 网络请求失败: {e}")
            return None
        except json.JSONDecodeError:
            print("[错误] 响应不是有效的JSON格式")
            return None

    def check_cookie_valid(self) -> Tuple[bool, str]:
        """验证Cookie有效性"""
        url = "https://api.bilibili.com/x/web-interface/nav"
        data = self.make_api_request(url)

        if not data:
            return False, "无法验证Cookie状态"

        if data.get('code') != 0:
            message = data.get('message', '未知错误')
            return False, f"Cookie无效: {message}"

        user_info = data.get('data', {})
        return True, f"登录用户: {user_info.get('uname', '未知用户')} (UID: {user_info.get('mid', '未知')})"

    def get_user_favorites_via_web(self, target_uid: str) -> Optional[List[Dict]]:
        """
        通过网页爬取获取用户收藏夹列表
        返回格式与API保持一致：[{'id': str, 'title': str, 'media_count': int}]
        """
        url = f"https://space.bilibili.com/{target_uid}/favlist"
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            # 检查是否跳转到登录页
            if "passport.bilibili.com" in response.url:
                print("[错误] 需要登录才能查看此用户的收藏夹")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # 检查是否有"该用户未公开收藏夹"提示
            no_favs = soup.find(text=re.compile(r'该用户未公开收藏夹'))
            if no_favs:
                print(f"[提示] 用户 {target_uid} 未公开收藏夹")
                return []

            # 解析收藏夹列表
            fav_list = []
            for item in soup.select('.fav-list .fav-item'):
                try:
                    fav_id = item['data-fid']
                    title = item.select_one('.fav-title').get_text(strip=True)
                    count = int(item.select_one('.fav-num').get_text(strip=True))

                    fav_list.append({
                        'id': fav_id,
                        'title': title,
                        'media_count': count
                    })
                except Exception as e:
                    print(f"[警告] 解析收藏夹条目时出错: {e}")
                    continue

            return fav_list

        except Exception as e:
            print(f"[错误] 网页爬取失败: {e}")
            return None

    def get_user_favorites(self, target_uid: str, fallback_to_web: bool = True) -> Optional[List[Dict]]:
        """
        获取用户收藏夹列表（优先API，失败后可选网页爬取）
        :param fallback_to_web: 当API失败时是否尝试网页爬取
        """
        # 优先尝试API
        api_result = self._get_user_favorites_api(target_uid)
        if api_result is not None:
            return api_result

        # API失败且允许回退到网页爬取
        if fallback_to_web:
            print("[提示] API获取失败，尝试网页爬取...")
            return self.get_user_favorites_via_web(target_uid)

        return None

    def _get_user_favorites_api(self, target_uid: str) -> Optional[List[Dict]]:
        """原始API方法（改名为私有方法）"""
        url = "https://api.bilibili.com/x/v3/fav/folder/created/list-all"
        params = {'up_mid': target_uid}

        data = self.make_api_request(url, params)
        if not data:
            return None

        if data.get('code') != 0:
            return None

        # 处理data为null的情况
        if data.get('data') is None:
            return []

        return data['data'].get('list', [])

    def get_favorite_contents(self, media_id: str, fav_title: str, page: int = 1) -> List[Dict]:
        """获取收藏夹内容（分页）"""
        url = "https://api.bilibili.com/x/v3/fav/resource/list"
        params = {
            'media_id': media_id,
            'pn': page,
            'ps': 20
        }

        data = self.make_api_request(url, params)
        if not data or data.get('code') != 0:
            print(f"[错误] 获取收藏夹内容失败: {fav_title}")
            return []

        # 添加收藏夹标题到每个视频
        videos = data.get('data', {}).get('medias', [])
        for video in videos:
            video['fav_title'] = fav_title
            video['fav_id'] = media_id

        return videos

    def get_all_favorite_contents(self, media_id: str, fav_title: str, max_videos: int = 7) -> List[Dict]:
        """获取收藏夹所有内容，但最多返回max_videos个视频"""
        all_videos = []
        page = 1
        max_retry = 3  # 最大重试次数
        random_value = random.randint(max_videos - 3, max_videos)
        while len(all_videos) < random_value:  # 修改循环条件
            try:
                videos = self.get_favorite_contents(media_id, fav_title, page)
                if not videos:
                    break

                # 计算还需要多少视频
                remaining = max_videos - len(all_videos)
                if len(videos) > remaining:
                    videos = videos[:remaining]  # 只取需要的数量

                all_videos.extend(videos)
                print(
                    f"[进度] 收藏夹 '{fav_title}' - 第{page}页: 获取{len(videos)}个视频 (总计: {len(all_videos)}/{max_videos})")

                # 检查是否还有更多
                if len(videos) < 20 or len(all_videos) >= remaining:
                    break

                page += 1
                time.sleep(1)  # 礼貌性延迟

            except Exception as e:
                print(f"[错误] 获取第{page}页失败: {e}")
                max_retry -= 1
                if max_retry <= 0:
                    break
                time.sleep(1)

        print(f"[完成] 收藏夹 '{fav_title}': 共获取{len(all_videos)}个视频 (最多{max_videos}个)")
        return all_videos

    def get_all_favorite_contents_for_user(self, favorites: List[Dict], max_videos: int = 7) -> List[Dict]:
        """获取用户所有收藏夹的内容，直到达到max_videos数量或遍历完所有收藏夹"""
        all_videos = []

        for fav in favorites:
            if len(all_videos) >= max_videos:
                break

            fav_id = fav['id']
            fav_title = fav['title']

            # 计算还需要多少视频
            remaining = max_videos - len(all_videos)

            # 获取当前收藏夹的内容，最多取remaining个
            contents = self.get_all_favorite_contents(fav_id, fav_title, max_videos=remaining)
            all_videos.extend(contents)

            # 如果已经达到数量要求，提前结束
            if len(all_videos) >= max_videos:
                break

            time.sleep(3)  # 收藏夹间延迟

        print(f"[完成] 用户总计获取 {len(all_videos)} 个视频 (最多{max_videos}个)")
        return all_videos

    def export_results(self, data: List[Dict], uid: str, format: str = "json"):
        """导出结果到文件"""
        timestamp = int(time.time())
        dir_name = f"bili_exports/{uid}_{timestamp}"
        os.makedirs(dir_name, exist_ok=True)

        # 分组保存
        fav_contents = {}
        for item in data:
            fav_id = item['fav_id']
            if fav_id not in fav_contents:
                fav_contents[fav_id] = []
            fav_contents[fav_id].append(item)

        # 为每个收藏夹单独保存
        all_files = []
        for fav_id, videos in fav_contents.items():
            # 获取收藏夹标题
            fav_title = videos[0]['fav_title'] if videos else "未知"

            # 安全文件名
            safe_title = "".join([c for c in fav_title if c.isalnum() or c in ' _-'])
            safe_title = safe_title[:50]

            if format == "json":
                filename = f"{safe_title}_{fav_id}.json"
                filepath = os.path.join(dir_name, filename)
                with open(filepath, 'w', encoding='utf-8-sig') as f:
                    json.dump(videos, f, ensure_ascii=False, indent=2)
            elif format == "csv":
                filename = f"{safe_title}_{fav_id}.csv"
                filepath = os.path.join(dir_name, filename)
                self._save_as_csv(videos, filepath)

            all_files.append(filepath)

        # 保存汇总文件
        if format == "json":
            summary_file = os.path.join(dir_name, f"summary_{uid}.json")
            with open(summary_file, 'w', encoding='utf-8-sig') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            all_files.append(summary_file)
            print(f"[导出] 数据已保存到 {summary_file}")

        # 压缩整个目录
        import shutil
        shutil.make_archive(dir_name, 'zip', dir_name)
        print(f"[完成] 结果已打包: {dir_name}.zip")

        return all_files

    def _save_as_csv(self, videos: List[Dict], filepath: str):
        """保存为CSV格式"""
        import csv

        if not videos:
            return

        # 获取所有可能的字段
        fields = set()
        for video in videos:
            fields.update(video.keys())
        fields = sorted(fields)

        with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(videos)

    def process_uid_range(self, uid_range: Tuple[int, int], cookies: str, export_format: str, max_videos: int = 7):
        """处理指定UID区间的用户"""
        local_results = []
        processed_uids = set()  # 记录已经处理过的UID
        failed_uids = set()  # 记录失败的UID
        success_count = 0
        max_attempts = 99999  # 最大尝试次数
        target_success = 1020  # 每个区间目标成功数

        # 初始化本地爬虫实例
        local_spider = BilibiliFavoritesSpider(cookies)

        attempts = 0
        while success_count < target_success and attempts < max_attempts:
            attempts += 1

            # 生成该区间内的随机UID，确保不重复
            while True:
                uid = random.randint(uid_range[0], uid_range[1])
                if uid not in processed_uids and uid not in failed_uids:
                    break

            processed_uids.add(uid)
            print(f"\n处理UID区间 {uid_range[0]}-{uid_range[1]}: 尝试第{attempts}次 (UID: {uid})")

            try:
                # 获取用户收藏夹列表
                favorites = local_spider.get_user_favorites(str(uid))

                # 检查是否成功获取收藏夹
                if favorites is None:
                    print(f"无法获取用户 {uid} 的收藏夹信息，可能是网络问题")
                    failed_uids.add(uid)
                    time.sleep(3)
                    continue
                elif not favorites:
                    print(f"用户 {uid} 没有可访问的收藏夹")
                    failed_uids.add(uid)
                    time.sleep(3)
                    continue

                # 成功获取到收藏夹
                print(f"成功获取用户 {uid} 的收藏夹列表")

                # 获取收藏夹内容
                contents = local_spider.get_all_favorite_contents_for_user(favorites, max_videos=max_videos)

                if contents:
                    local_results.extend(contents)
                    success_count += 1

                    # 导出当前用户的结果
                    local_spider.export_results(contents, str(uid), export_format)
                    print(f"成功处理用户 {uid} (当前成功数: {success_count}/{target_success})")
                else:
                    print(f"用户 {uid} 没有可获取的视频内容")
                    failed_uids.add(uid)

                time.sleep(3)  # 间隔时间

            except Exception as e:
                print(f"处理用户 {uid} 时发生错误: {str(e)}")
                failed_uids.add(uid)
                time.sleep(3)

        print(f"\nUID区间 {uid_range[0]}-{uid_range[1]} 处理完成: "
              f"成功 {success_count} 个, 失败 {len(failed_uids)} 个, 总尝试 {attempts} 次")

        return local_results

def get_known_active_uids():
    """获取已知活跃用户UID"""
    return [
        2,  # B站官方账号
        546195,  # 老番茄
        883968,  # 罗翔说刑法
        927587,  # 何同学
        777536,  # 绵羊料理
        125930,  # 敖厂长
        362356,  # 李子柒
        282994,  # 手工耿
        158937764,  # 新用户示例
        123456789  # 随机大数字
    ]

# 修改main函数
def main():
    print("=" * 60)
    print("B站用户收藏夹批量下载工具 - 多进程优化版")
    print("=" * 60)

    # 询问是否使用Cookie
    use_cookie = input("\n是否需要使用Cookie? (y/n): ").lower() == 'y'
    cookies = None
    if use_cookie:
        print("\n" + "=" * 60)
        print("请按以下步骤获取Cookie:")
        print("1. 登录Bilibili")
        print("2. 按F12打开开发者工具")
        print("3. 刷新页面")
        print("4. 点击Network标签，然后点击任意请求")
        print("5. 在Headers中找到'Cookie: '开头的完整内容")
        print("6. 复制后粘贴到这里")
        print("=" * 60)
        cookies = input("粘贴你的Cookie: ").strip()

    # 询问导出格式
    print("\n导出选项:")
    print("1. JSON格式")
    print("2. CSV格式")
    export_choice = input("选择导出格式 (1/2): ").strip()
    export_format = "json" if export_choice == "1" else "csv"

    # 划分UID区间 (1-100000000分为5个区间)
    uid_ranges = []
    range_size = 100000000 // 10
    for i in range(10):
        start = i * range_size + 1
        end = (i + 1) * range_size if i < 9 else 100000000
        uid_ranges.append((start, end))

    print("\nUID区间划分:")
    for i, (start, end) in enumerate(uid_ranges, 1):
        print(f"{i}. {start:,} - {end:,}")

        # 创建进程池
        num_processes = min(10, multiprocessing.cpu_count())
        print(f"\n启动 {num_processes} 个进程并行处理...")

        # 初始化爬虫实例
        spider = BilibiliFavoritesSpider(cookies)

        # 验证Cookie有效性
        if cookies:
            valid, message = spider.check_cookie_valid()
            print(message)
            if not valid:
                print("使用无Cookie模式继续...")

        # 准备多进程参数
        process_func = partial(
            spider.process_uid_range,
            cookies=cookies,
            export_format=export_format,
            max_videos=7
        )

        # 启动多进程
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 使用imap_unordered获取实时进度
            all_results = []
            for result in pool.imap_unordered(process_func, uid_ranges):
                all_results.extend(result)
                print(f"\n收到一个区间处理结果，当前总计获取 {len(all_results)} 个视频")

    # 合并所有结果
    final_results = []
    for result in all_results:
        final_results.extend(result)

    # 导出汇总结果
    if final_results:
        print("\n正在导出所有用户的汇总结果...")
        timestamp = int(time.time())
        summary_dir = f"bili_exports/summary_{timestamp}"
        os.makedirs(summary_dir, exist_ok=True)

        if export_format == "json":
            summary_file = os.path.join(summary_dir, "all_users_summary.json")
            with open(summary_file, 'w', encoding='utf-8-sig') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
        else:
            summary_file = os.path.join(summary_dir, "all_users_summary.csv")
            spider._save_as_csv(final_results, summary_file)

        print(f"\n[完成] 所有用户数据已汇总保存到: {summary_file}")

    print("\n处理完成!")


if __name__ == "__main__":
    # Windows系统需要添加这行代码
    multiprocessing.freeze_support()
    main()