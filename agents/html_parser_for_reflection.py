"""
HTML Parser for Reflection
解析生成的HTML文件，生成用于AI评估的序列化格式

支持两种类型：
1. ITProduct (小红书图文): image_text.html
2. CommentProduct (虎扑讨论): discussion_post.html

输出格式示例：
{text_0: "第一段文字内容..."}
{image_0: "personalized_post_1.png"}  # 图片路径，供多模态模型单独分析
{text_1: "第二段文字内容..."}
{link_0: "链接标题 | https://..."}

注意：图片只保存路径，实际分析由多模态模型（GPT-4 Vision等）单独处理
"""

import os
import base64
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup


class HTMLParserForReflection:
    """解析HTML为AI可读的序列格式"""
    
    def __init__(self):
        pass
    
    def parse_html_to_sequence(self, html_path: str) -> Dict:
        """
        解析HTML文件，生成序列化文本和图片路径列表
        
        Args:
            html_path: HTML文件路径
            
        Returns:
            {
                "sequence_text": "序列化的文本字符串",
                "image_paths": ["绝对路径1", "绝对路径2", ...],
                "stats": {"texts": N, "images": M, "links": K}
            }
        """
        html_path = Path(html_path)
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_path}")
        
        # 读取HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # 提取内容序列
        content_div = soup.find('div', class_='post-content')
        if not content_div:
            return {
                "sequence_text": "Error: Could not find post-content div",
                "image_paths": [],
                "stats": {"texts": 0, "images": 0, "links": 0}
            }
        
        # 生成序列
        sequence_parts = []
        image_paths = []  # 保存图片的绝对路径
        text_idx = 0
        image_idx = 0
        link_idx = 0
        
        base_dir = html_path.parent
        
        for element in content_div.children:
            # 1. 文本段落
            if element.name == 'p' or element.name == 'h2':
                text_content = element.get_text(strip=True)
                if text_content:
                    # 清理markdown标记
                    text_content = self._clean_markdown(text_content)
                    sequence_parts.append(f"{{text_{text_idx}: \"{text_content}\"}}")
                    text_idx += 1
            
            # 2. 图片（只保存路径，不编码base64）
            elif element.name == 'div' and 'post-image' in element.get('class', []):
                img_tag = element.find('img')
                if img_tag and img_tag.get('src'):
                    img_src = img_tag['src']
                    img_path = base_dir / img_src
                    
                    if img_path.exists():
                        # 只保存相对路径到序列文本
                        sequence_parts.append(f"{{image_{image_idx}: \"{img_src}\"}}")
                        # 保存绝对路径供多模态模型使用
                        image_paths.append(str(img_path.absolute()))
                        image_idx += 1
                    else:
                        sequence_parts.append(f"{{image_{image_idx}: \"[图片不存在: {img_src}]\"}}")
                        image_paths.append(None)
                        image_idx += 1
            
            # 3. 链接
            elif element.name == 'a' and 'link-card' in element.get('class', []):
                title_tag = element.find('div', class_='link-title')
                title = title_tag.get_text(strip=True) if title_tag else "未知链接"
                
                url = element.get('href', '#')
                
                link_text = f"{title} | {url}"
                sequence_parts.append(f"{{link_{link_idx}: \"{link_text}\"}}")
                link_idx += 1
        
        # 拼接结果
        sequence_text = "\n".join(sequence_parts)
        
        # 添加统计信息
        stats_line = f"\n\n[Statistics: {text_idx} texts, {image_idx} images, {link_idx} links]"
        sequence_text += stats_line
        
        return {
            "sequence_text": sequence_text,
            "image_paths": image_paths,
            "stats": {
                "texts": text_idx,
                "images": image_idx,
                "links": link_idx
            }
        }
    
    def _clean_markdown(self, text: str) -> str:
        """清理文本中的markdown标记"""
        import re
        # 移除加粗标记 **text**
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        # 移除其他markdown标记（如果有）
        text = re.sub(r'__', '', text)
        return text


def test_parser():
    """测试解析器"""
    import sys
    
    print("="*80)
    print("🧪 测试 HTML Parser for Reflection")
    print("="*80)
    
    parser = HTMLParserForReflection()
    
    # 测试文件
    test_files = [
        ("generated_it/29_5726fe0950c4b401f76283be/image_text.html", "ITProduct"),
        ("generated_posts/3_132887808576185/discussion_post.html", "CommentProduct")
    ]
    
    for html_path, post_type in test_files:
        if not os.path.exists(html_path):
            print(f"\n⚠️  文件不存在: {html_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"📝 测试: {post_type}")
        print(f"   文件: {html_path}")
        print(f"{'='*80}")
        
        try:
            # 解析HTML（只保存图片路径，不含base64）
            print("\n生成AI Reflection输入格式（图片仅保存路径）")
            print("-"*80)
            result = parser.parse_html_to_sequence(html_path)
            
            sequence_text = result["sequence_text"]
            image_paths = result["image_paths"]
            stats = result["stats"]
            
            # 只打印前1000字符用于预览
            print(sequence_text[:1000])
            if len(sequence_text) > 1000:
                print(f"\n... (总共 {len(sequence_text)} 字符，已截断)")
            
            # 统计信息
            print(f"\n📊 统计: {stats['texts']} 文本, {stats['images']} 图片, {stats['links']} 链接")
            
            # 图片路径列表
            if image_paths:
                print(f"\n🖼️  图片路径列表:")
                for i, img_path in enumerate(image_paths):
                    if img_path:
                        print(f"   {i}. {img_path}")
                    else:
                        print(f"   {i}. [图片缺失]")
            
        except Exception as e:
            print(f"❌ 解析失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("✅ 测试完成")
    print("="*80)


if __name__ == "__main__":
    test_parser()

