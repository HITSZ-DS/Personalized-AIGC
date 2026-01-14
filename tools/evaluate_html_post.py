"""
HTML Post Evaluator with Visual Support

This tool evaluates AI-generated social media posts (HTML format) by:
1. Extracting images from HTML
2. Generating screenshots of the rendered HTML
3. Providing all visual content to AI for comprehensive evaluation
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
from PIL import Image
import io

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: playwright not available. Screenshot generation will be disabled.")


class HTMLPostExtractor:
    """Extract content and images from HTML posts"""
    
    def __init__(self, html_path: str):
        self.html_path = Path(html_path)
        self.base_dir = self.html_path.parent
        
        with open(html_path, 'r', encoding='utf-8') as f:
            self.soup = BeautifulSoup(f.read(), 'html.parser')
    
    def extract_images(self) -> List[Dict[str, str]]:
        """Extract all image paths from HTML"""
        images = []
        image_divs = self.soup.find_all('div', class_='post-image')
        
        for img_div in image_divs:
            img_tag = img_div.find('img')
            if not img_tag:
                continue
            
            img_src = img_tag.get('src', '')
            if not img_src:
                continue
            
            image_path = self.base_dir / img_src
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Get caption
            caption_div = img_div.find('div', class_='image-caption')
            caption = caption_div.get_text(strip=True) if caption_div else ""
            
            images.append({
                'path': str(image_path),
                'src': img_src,
                'caption': caption
            })
        
        return images
    
    def extract_links(self) -> List[Dict[str, str]]:
        """Extract all link cards from HTML"""
        links = []
        link_cards = self.soup.find_all('a', class_='link-card')
        
        for link_card in link_cards:
            href = link_card.get('href', '')
            platform_tag = link_card.find('span', class_='link-platform-tag')
            title_div = link_card.find('div', class_='link-title')
            
            links.append({
                'url': href,
                'platform': platform_tag.get_text(strip=True) if platform_tag else "",
                'title': title_div.get_text(strip=True) if title_div else ""
            })
        
        return links
    
    def extract_text_content(self) -> Dict[str, str]:
        """Extract text content from HTML"""
        content = {}
        
        # Extract tags (小红书风格)
        tags = []
        tag_div = self.soup.find('div', class_='post-tags')
        if tag_div:
            for tag_span in tag_div.find_all('span', class_='tag'):
                tags.append(tag_span.get_text(strip=True))
        
        # Extract hot topic tags (虎扑风格)
        hot_tags = []
        footer = self.soup.find('div', class_='footer')
        if footer:
            for hot_tag in footer.find_all('span', class_='hot-topic-tag'):
                hot_tags.append(hot_tag.get_text(strip=True))
        
        content['tags'] = tags if tags else hot_tags
        
        # Extract main content
        content_div = self.soup.find('div', class_='post-content')
        if content_div:
            content['main_text'] = self._clean_text(content_div.get_text())
            # Extract title if exists (hupu style)
            h2 = content_div.find('h2')
            if h2:
                content['title'] = h2.get_text(strip=True)
            else:
                content['title'] = ""
        else:
            content['main_text'] = ""
            content['title'] = ""
        
        # Extract header info
        header = self.soup.find('div', class_='post-header')
        if header:
            user_info = header.find('div', class_='user-info')
            if user_info:
                h3 = user_info.find('h3')
                content['author'] = h3.get_text(strip=True) if h3 else ""
                time_div = user_info.find('div', class_='post-time')
                content['post_time'] = time_div.get_text(strip=True) if time_div else ""
        
        return content
    
    def detect_post_type(self) -> str:
        """
        Detect post type: 'redbook' or 'hupu'
        
        Returns:
            'redbook' or 'hupu'
        """
        # Check by path
        html_path_str = str(self.html_path)
        if 'hupu' in html_path_str.lower() or 'discussion_post' in html_path_str.lower():
            return 'hupu'
        if 'redbook' in html_path_str.lower() or 'image_text' in html_path_str.lower():
            return 'redbook'
        
        # Check by content features
        # Hupu: has links, no images (or very few), has hot topic tags
        # Redbook: has images, has hashtags
        images = self.extract_images()
        links = self.extract_links()
        footer = self.soup.find('div', class_='footer')
        has_hot_tags = footer and footer.find('span', class_='hot-topic-tag')
        
        if len(links) > 0 and len(images) == 0 and has_hot_tags:
            return 'hupu'
        if len(images) > 0:
            return 'redbook'
        
        # Default to redbook if uncertain
        return 'redbook'
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_css_info(self) -> Dict[str, any]:
        """Extract CSS styling information"""
        styles = {}
        style_tag = self.soup.find('style')
        if style_tag:
            styles['has_custom_styles'] = True
            # Extract key style information
            style_text = style_tag.get_text()
            styles['has_gradient'] = 'gradient' in style_text
            styles['has_shadow'] = 'shadow' in style_text
            styles['has_border_radius'] = 'border-radius' in style_text
        else:
            styles['has_custom_styles'] = False
        
        return styles


class HTMLScreenshotGenerator:
    """Generate screenshots of rendered HTML"""
    
    def __init__(self):
        self.playwright_available = PLAYWRIGHT_AVAILABLE
    
    def generate_screenshot(self, html_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate a screenshot of the rendered HTML
        
        Args:
            html_path: Path to HTML file
            output_path: Optional path to save screenshot. If None, saves next to HTML file.
        
        Returns:
            Path to saved screenshot, or None if failed
        """
        if not self.playwright_available:
            print("Playwright not available. Install with: pip install playwright && playwright install chromium")
            return None
        
        html_path = Path(html_path)
        if output_path is None:
            output_path = html_path.parent / f"{html_path.stem}_screenshot.png"
        else:
            output_path = Path(output_path)
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={'width': 800, 'height': 1200})
                
                # Load HTML file
                page.goto(f"file://{html_path.absolute()}")
                
                # Wait for images to load
                page.wait_for_timeout(2000)
                
                # Take screenshot
                page.screenshot(path=str(output_path), full_page=True)
                browser.close()
            
            return str(output_path)
        except Exception as e:
            print(f"Error generating screenshot: {e}")
            return None
    
    def image_to_base64(self, image_path: str) -> Optional[str]:
        """Convert image to base64 string for API transmission"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_str = base64.b64encode(image_data).decode('utf-8')
                # Determine image format
                img = Image.open(io.BytesIO(image_data))
                format_map = {'PNG': 'png', 'JPEG': 'jpeg', 'JPG': 'jpeg'}
                img_format = format_map.get(img.format, 'png')
                return f"data:image/{img_format};base64,{base64_str}"
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None


class PostEvaluationPackage:
    """Package all information for AI evaluation"""
    
    def __init__(self, html_path: str, generate_screenshot: bool = True):
        self.html_path = Path(html_path)
        self.extractor = HTMLPostExtractor(html_path)
        self.screenshot_gen = HTMLScreenshotGenerator() if generate_screenshot else None
        
        # Extract content
        self.extractor = HTMLPostExtractor(html_path)
        self.images = self.extractor.extract_images()
        self.text_content = self.extractor.extract_text_content()
        self.css_info = self.extractor.get_css_info()
        self.links = self.extractor.extract_links()
        self.post_type = self.extractor.detect_post_type()
        self.screenshot_path = None
        
        # Generate screenshot if requested
        if generate_screenshot and self.screenshot_gen:
            self.screenshot_path = self.screenshot_gen.generate_screenshot(html_path)
    
    def to_dict(self, include_image_data: bool = False) -> Dict:
        """
        Convert to dictionary for AI evaluation
        
        Args:
            include_image_data: If True, include base64-encoded images. 
                               If False, only include image paths.
        """
        result = {
            'html_path': str(self.html_path),
            'post_type': self.detect_post_type(),
            'text_content': self.text_content,
            'css_info': self.css_info,
            'image_count': len(self.images),
            'images': [],
            'link_count': len(self.links),
            'links': self.links
        }
        
        # Add image information
        for img_info in self.images:
            img_data = {
                'path': img_info['path'],
                'src': img_info['src'],
                'caption': img_info['caption']
            }
            
            if include_image_data and self.screenshot_gen:
                base64_data = self.screenshot_gen.image_to_base64(img_info['path'])
                if base64_data:
                    img_data['base64_data'] = base64_data
            
            result['images'].append(img_data)
        
        # Add screenshot info
        if self.screenshot_path:
            result['screenshot_path'] = self.screenshot_path
            if include_image_data and self.screenshot_gen:
                screenshot_base64 = self.screenshot_gen.image_to_base64(self.screenshot_path)
                if screenshot_base64:
                    result['screenshot_base64'] = screenshot_base64
        
        return result
    
    def to_prompt_format(self) -> str:
        """Format as a prompt-friendly string for AI evaluation"""
        lines = []
        lines.append("=" * 60)
        lines.append("HTML POST EVALUATION PACKAGE")
        lines.append("=" * 60)
        lines.append(f"\nHTML File: {self.html_path.name}")
        lines.append(f"\n--- Text Content ---")
        lines.append(f"Author: {self.text_content.get('author', 'N/A')}")
        lines.append(f"Post Time: {self.text_content.get('post_time', 'N/A')}")
        lines.append(f"Tags: {', '.join(self.text_content.get('tags', []))}")
        lines.append(f"\nMain Text:\n{self.text_content.get('main_text', '')}")
        
        lines.append(f"\n--- Visual Content ---")
        lines.append(f"Number of Images: {len(self.images)}")
        for i, img in enumerate(self.images, 1):
            lines.append(f"\nImage {i}:")
            lines.append(f"  Path: {img['path']}")
            lines.append(f"  Caption: {img['caption']}")
        
        lines.append(f"\n--- Links ---")
        lines.append(f"Number of Links: {len(self.links)}")
        for i, link in enumerate(self.links, 1):
            lines.append(f"\nLink {i}:")
            lines.append(f"  Platform: {link['platform']}")
            lines.append(f"  Title: {link['title']}")
            lines.append(f"  URL: {link['url']}")
        
        lines.append(f"\n--- Post Type ---")
        lines.append(f"Detected Type: {self.post_type}")
        
        lines.append(f"\n--- Styling Info ---")
        lines.append(f"Custom Styles: {self.css_info.get('has_custom_styles', False)}")
        lines.append(f"Has Gradients: {self.css_info.get('has_gradient', False)}")
        lines.append(f"Has Shadows: {self.css_info.get('has_shadow', False)}")
        
        if self.screenshot_path:
            lines.append(f"\nScreenshot: {self.screenshot_path}")
            lines.append("(You can view the rendered HTML layout in the screenshot)")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def evaluate_html_post(html_path: str, generate_screenshot: bool = True) -> PostEvaluationPackage:
    """
    Main function to prepare HTML post for AI evaluation
    
    Args:
        html_path: Path to HTML file
        generate_screenshot: Whether to generate screenshot of rendered HTML
    
    Returns:
        PostEvaluationPackage object
    """
    return PostEvaluationPackage(html_path, generate_screenshot)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_html_post.py <html_path> [--no-screenshot]")
        sys.exit(1)
    
    html_path = sys.argv[1]
    generate_screenshot = "--no-screenshot" not in sys.argv
    
    package = evaluate_html_post(html_path, generate_screenshot)
    
    # Print prompt format
    print(package.to_prompt_format())
    
    # Save as JSON
    output_json = Path(html_path).parent / f"{Path(html_path).stem}_evaluation_package.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(package.to_dict(include_image_data=False), f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Evaluation package saved to: {output_json}")

