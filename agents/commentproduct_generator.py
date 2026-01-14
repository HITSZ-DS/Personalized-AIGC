import base64
import os
import json
import requests
import urllib.parse
import re
import random
import glob
from datetime import datetime
from pathlib import Path
import http.client

# Handle imports that work both from main.py (absolute) and from agents/ directory (relative)
try:
    # Try absolute import first (when running from main.py)
    from agents.rag_embedding_helper import RAGEmbeddingHelper
except ImportError:
    # Fallback to relative import (when running from agents/ directory)
    from rag_embedding_helper import RAGEmbeddingHelper

# Reflection mechanism imports
try:
    try:
        from agents.evaluate_groupscore import TextEmbeddingEvaluator, evaluate_file_links
    except ImportError:
        from evaluate_groupscore import TextEmbeddingEvaluator, evaluate_file_links
    REFLECTION_AVAILABLE = True
except ImportError:
    REFLECTION_AVAILABLE = False
    print("⚠️  Reflection mechanism not available (evaluate_groupscore not found)")


class CommentProductGenerator:
    def __init__(self):
        # Text generation model
        self.chat_api_key = os.getenv("CHAT_API_KEY")
        self.chat_base_url = os.getenv("CHAT_BASE_URL")
        self.chat_model = os.getenv("CHAT_MODEL")

        # Image generation model
        self.generate_api_key = os.getenv("GENERATE_API_KEY")
        self.generate_base_url = os.getenv("GENERATE_BASE_URL")
        self.generate_model = os.getenv("GENERATE_MODEL")
        
        # Search model for hot topic discovery
        self.search_api_key = os.getenv("SEARCH_API_KEY")
        self.search_base_url = os.getenv("SEARCH_BASE_URL")
        self.search_model = os.getenv("SEARCH_MODEL")
        
        # Examples directory
        self.examples_dir = os.path.join(os.path.dirname(__file__), "hupu")
        
        # Cache for example posts (load once, reuse across multiple generations)
        self._examples_cache = None
        
        # RAG settings
        self.rag_enabled = True  # Enable RAG-based example retrieval
        self.dataset_name = "hupu"  # Dataset name for cache identification
        self.data_root = os.path.join(os.path.dirname(__file__), "..", "download", "hupu")
        self._rag_cache = {}  # Cache for RAG retrieval results (key: user_id)
        
        # Initialize RAG embedding helper
        self.rag_helper = RAGEmbeddingHelper(
            api_key=self.search_api_key,
            api_base=self.search_base_url,
            cache_dir=os.path.join(os.path.dirname(__file__), "..", "embeddings_cache")
        )
        
        # Reflection mechanism settings
        self.reflection_enabled = REFLECTION_AVAILABLE and True  # Enable if available
        self.reflection_threshold = float(os.getenv("REFLECTION_THRESHOLD_HUPU", "0.75"))
        # Maximum reflection iterations (can be configured via environment variable)
        # First 3 iterations use specific strategies, iterations >= 3 all use iteration 2's strategy
        self.max_reflection_iterations = int(os.getenv("MAX_REFLECTION_ITERATIONS_HUPU", "10"))
        
        # Initialize reflection components (Link-Text relevance evaluator)
        if self.reflection_enabled:
            try:
                self.text_evaluator = TextEmbeddingEvaluator(
                    api_key=self.search_api_key,
                    api_base=self.search_base_url
                )
                print(f"✅ Reflection机制已启用 (虎扑链接-文本评估, 阈值: {self.reflection_threshold}, 最多{self.max_reflection_iterations}次迭代)")
                if self.max_reflection_iterations > 3:
                    print(f"   ℹ️  前3次使用特定策略，第4-{self.max_reflection_iterations}次重复使用第3次策略")
            except Exception as e:
                print(f"⚠️  Reflection机制初始化失败: {e}, 将跳过reflection")
                self.reflection_enabled = False
        
    def extract_top1_preference(self, profile_data):
        """
        Extract top1 preference from user profile
        
        Args:
            profile_data: User profile dict or string
            
        Returns:
            String containing the top1 preference description
        """
        if isinstance(profile_data, dict):
            profile_text = profile_data.get("profile_text", json.dumps(profile_data, ensure_ascii=False))
        else:
            profile_text = str(profile_data)
        
        # Try to extract "1. Preference 1:" or "Preference 1:" pattern
        patterns = [
            r'1\.\s*Preference\s*1:\s*([^\n]+(?:\n\s+Reason:[^\n]+)?)',
            r'Preference\s*1:\s*([^\n]+(?:\n\s+Reason:[^\n]+)?)',
            r'1\.\s*([^\n]+(?:\n\s+Reason:[^\n]+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, profile_text, re.IGNORECASE)
            if match:
                preference = match.group(1).strip()
                print(f"📋 Extracted Top1 Preference: {preference[:100]}...")
                return preference
        
        # Fallback: return first 500 chars of profile
        print(f"⚠️ Could not extract structured preference, using profile preview")
        return profile_text[:500]
    
    def extract_user_id_from_path(self, file_path):
        """
        Extract user ID from file path
        
        Args:
            file_path: Path to user profile or output directory
            
        Returns:
            User ID string, or None if not found
        """
        # Try to extract from path pattern like: generated_it/0_5435e123d6e4a965e190095a/
        # or download/hupu/132349263326575/
        
        # Pattern 1: {number}_{alphanumeric_id}
        match = re.search(r'[\\/](\d+_[a-f0-9]+)[\\/]', file_path)
        if match:
            full_id = match.group(1)
            # Extract the alphanumeric part after underscore
            user_id = full_id.split('_', 1)[1] if '_' in full_id else full_id
            print(f"📌 Extracted user_id from path: {user_id}")
            return user_id
        
        # Pattern 2: pure numeric ID
        match = re.search(r'[\\/](\d{10,})[\\/]', file_path)
        if match:
            user_id = match.group(1)
            print(f"📌 Extracted user_id from path: {user_id}")
            return user_id
        
        print(f"⚠️ Could not extract user_id from path: {file_path}")
        return None
    
    def load_examples_with_rag(self, user_id, top1_preference, top_k=3):
        """
        Load examples using RAG (Retrieval-Augmented Generation)
        Retrieves most relevant examples from user's historical and recommended posts
        
        Args:
            user_id: User ID
            top1_preference: Top1 preference text for similarity matching
            top_k: Number of examples to retrieve
            
        Returns:
            List of example dicts with title and content parsed
        """
        # Check cache
        cache_key = f"{user_id}_{top_k}"
        if cache_key in self._rag_cache:
            print(f"✅ Using cached RAG examples ({len(self._rag_cache[cache_key])} posts)")
            return self._rag_cache[cache_key]
        
        print(f"🔍 RAG Mode: Retrieving relevant examples for user {user_id}...")
        
        try:
            # Step 1: Build embeddings for user files
            print(f"📊 Building embeddings for user files...")
            embeddings_data = self.rag_helper.build_embeddings_for_user(
                dataset_name=self.dataset_name,
                dataset_root=self.data_root,
                user_id=user_id,
                max_workers=10,
                use_cache=True
            )
            
            if not embeddings_data["embeddings"]:
                print(f"⚠️ No embeddings found, falling back to default examples")
                return self.load_examples_fallback()
            
            # Step 2: Retrieve top-k similar files
            print(f"🎯 Retrieving top-{top_k} similar examples based on preference...")
            similar_files = self.rag_helper.retrieve_top_k_similar(
                query_text=top1_preference,
                embeddings_data=embeddings_data,
                top_k=top_k
            )
            
            if not similar_files:
                print(f"⚠️ No similar files found, falling back to default examples")
                return self.load_examples_fallback()
            
            # Step 3: Load file contents and parse
            examples = []
            for i, file_info in enumerate(similar_files):
                try:
                    content = self.rag_helper.get_file_content(file_info["path"])
                    
                    if not content:
                        continue
                    
                    # Parse title and content
                    title = ""
                    parsed_content = content
                    
                    # Check if file has Title: and Content: format
                    if "Title:" in content and "Content:" in content:
                        lines = content.split('\n')
                        for j, line in enumerate(lines):
                            if line.startswith("Title:"):
                                title = line.replace("Title:", "").strip()
                            elif line.startswith("Content:"):
                                parsed_content = '\n'.join(lines[j:]).replace("Content:", "").strip()
                                break
                    
                    examples.append({
                        "filename": file_info["filename"],
                        "title": title,
                        "content": parsed_content,
                        "full_text": content,
                        "similarity": file_info["similarity"],
                        "folder": file_info["folder"],
                        "post_id": file_info["post_id"]
                    })
                    
                    print(f"   ✅ Retrieved: {file_info['filename']} (similarity: {file_info['similarity']:.3f})")
                    if title:
                        print(f"      标题: {title[:50]}...")
                        
                except Exception as e:
                    print(f"   ⚠️ Failed to load {file_info['path']}: {e}")
            
            # Cache the results
            self._rag_cache[cache_key] = examples
            print(f"💾 Cached {len(examples)} RAG example(s)")
            
            return examples
            
        except Exception as e:
            print(f"⚠️ RAG retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return self.load_examples_fallback()
    
    def load_examples_fallback(self):
        """
        Fallback method: Load examples from fixed directory (agents/hupu/)
        This is the original load_examples logic
        """
        examples = []
        
        if not os.path.exists(self.examples_dir):
            print(f"⚠️ Examples directory not found: {self.examples_dir}")
            return examples
        
        # Find all .txt files in the examples directory
        txt_files = glob.glob(os.path.join(self.examples_dir, "*.txt"))
        
        if not txt_files:
            print(f"⚠️ No example files found in {self.examples_dir}")
            return examples
        
        print(f"📚 Loading {len(txt_files)} fallback example(s) from {self.examples_dir}...")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    raw_content = f.read().strip()
                    
                    if not raw_content:
                        continue
                    
                    # Parse title and content
                    title = ""
                    content = raw_content
                    
                    # Check if file has Title: and Content: format
                    if "Title:" in raw_content and "Content:" in raw_content:
                        lines = raw_content.split('\n')
                        for i, line in enumerate(lines):
                            if line.startswith("Title:"):
                                title = line.replace("Title:", "").strip()
                            elif line.startswith("Content:"):
                                content = '\n'.join(lines[i:]).replace("Content:", "").strip()
                                break
                    
                    examples.append({
                        "filename": os.path.basename(txt_file),
                        "title": title,
                        "content": content,
                        "full_text": raw_content
                    })
                    print(f"   ✅ Loaded: {os.path.basename(txt_file)} ({len(content)} chars)")
                    if title:
                        print(f"      标题: {title[:50]}...")
                        
            except Exception as e:
                print(f"   ⚠️ Failed to load {txt_file}: {e}")
        
        return examples
    
    def load_examples(self, user_profile_path=None, profile_data=None):
        """
        Load example posts - supports both RAG mode and fallback mode
        
        Args:
            user_profile_path: Path to user profile (for extracting user_id)
            profile_data: User profile data (for extracting top1 preference)
            
        Returns:
            List of example dicts with title and content parsed
        """
        # If RAG is enabled and we have necessary info, use RAG
        if self.rag_enabled and user_profile_path and profile_data:
            try:
                # Extract user_id from path
                user_id = self.extract_user_id_from_path(user_profile_path)
                
                if user_id:
                    # Extract top1 preference
                    top1_preference = self.extract_top1_preference(profile_data)
                    
                    # Use RAG to retrieve examples
                    examples = self.load_examples_with_rag(user_id, top1_preference, top_k=3)
                    
                    if examples:
                        return examples
                    else:
                        print("⚠️ RAG returned no examples, using fallback")
                else:
                    print("⚠️ Could not extract user_id, using fallback")
            except Exception as e:
                print(f"⚠️ RAG mode failed: {e}, using fallback")
                import traceback
                traceback.print_exc()
        
        # Fallback: Return cached examples if already loaded
        if self._examples_cache is not None:
            print(f"✅ Using cached fallback examples ({len(self._examples_cache)} posts)")
            return self._examples_cache
        
        # Use fallback method
        examples = self.load_examples_fallback()
        
        # Cache the loaded examples
        self._examples_cache = examples
        print(f"💾 Cached {len(examples)} fallback example(s) for future use")
        
        return examples
    
    def search_hot_topics(self, user_profile, retry_count=0, max_retries=2):
        """
        Real web-search-based hot topic discovery using gpt-5-all.
        Reference: https://yunwu.apifox.cn/api-306423418
        Key: Must include web_search_options: {} parameter!
        
        Args:
            user_profile: 用户画像
            retry_count: 当前重试次数
            max_retries: 最大重试次数
        """

        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M')

        # 改进：要求返回与用户兴趣相关的具体热点事件
        search_prompt = f"""基于以下用户画像，搜索当前（{current_time_str}）与用户兴趣相关的2-3个**具体热点事件**。

用户画像：
{user_profile[:500]}

返回JSON格式：
[
  {{
    "topic": "具体热点事件的简短描述（如：ZyWOo夺得CSGO Major FMVP）",
    "platform": "平台名称（优先中文平台如微博/知乎/虎扑/B站，实在没有可用英文平台）",
    "search_keyword": "精确搜索关键词（用于生成搜索链接，必填）",
    "url": "如果找到具体讨论/新闻链接则填写（选填）",
    "title": "如果有URL则填写标题（选填）"
  }}
]

**核心要求**：
1. **必须是具体的热点事件**，不能是泛泛的"热榜"或"热搜榜"
2. 热点应该与用户的兴趣领域相关（如用户关注体育，就找体育热点）
3. **search_keyword 是必填项**，要精确（如"ZyWOo FMVP"、"CBA 邱彪 禁赛"）
4. **链接优先级（按顺序尝试）**：
   - 第一优先：中文平台的讨论帖（微博、知乎、虎扑、B站、抖音、新浪体育、腾讯体育等）
   - 第二优先：中文新闻网站（新浪、网易、腾讯、搜狐等）
   - 降级方案：如果确实找不到合适的中文链接，英文权威媒体链接也可以（ESPN、BBC、华盛顿邮报等）
5. **标题语言应与链接语言一致**（中文链接用中文标题，英文链接用英文标题）
6. **确保每个热点都有有效的 search_keyword**
7. 如果真的找不到任何相关热点，才返回：NO_VERIFIED_TRENDS_FOUND"""

        try:
            print("🔍 Calling gpt-5-all with web_search_options...")
            print(f"📍 API Endpoint: https://yunwu.ai/v1/chat/completions")
            print(f"📍 Model: gpt-5-all")
            print(f"📍 Search Query Time: {current_time_str}")
            
            # 使用 gpt-5-all + web_search_options
            response = requests.post(
                "https://yunwu.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.search_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-5-all",
                    "web_search_options": {},  # 启用联网搜索
                    "messages": [
                        {
                            "role": "user",
                            "content": search_prompt
                        }
                    ],
                    "temperature": 0.7  # 添加温度参数使输出更可控
                },
                timeout=90  # 增加超时时间到90秒
            )
            
            print(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"⚠️ API Error {response.status_code}")
                print(f"⚠️ Response Text: {response.text[:500]}")
                return "NO_VERIFIED_TRENDS_FOUND"
            
            resp_json = response.json()
            
            # 保存原始响应用于调试
            debug_path = os.path.join(os.path.dirname(__file__), "..", "debug_search_response.json")
            try:
                with open(debug_path, 'w', encoding='utf-8') as f:
                    json.dump(resp_json, f, ensure_ascii=False, indent=2)
                print(f"🐛 Debug: Full API response saved to {debug_path}")
            except:
                pass
            
            if "choices" not in resp_json:
                print(f"⚠️ Unexpected response format (no 'choices' field)")
                print(f"⚠️ Response keys: {list(resp_json.keys())}")
                return "NO_VERIFIED_TRENDS_FOUND"
            
            content = resp_json["choices"][0]["message"]["content"]
            
            if not content:
                print("⚠️ Empty content from search")
                return "NO_VERIFIED_TRENDS_FOUND"
            
            # 🔍 调试：打印原始 content
            print(f"📄 Web Search Raw Response (first 300 chars):")
            print(f"{content[:300]}")
            print(f"...")
            
            # 检查明确的失败消息（但允许有JSON数组的情况）
            if "NO_VERIFIED_TRENDS_FOUND" in content and "[" not in content:
                print("⚠️ Web search explicitly failed (no JSON array found)")
                return "NO_VERIFIED_TRENDS_FOUND"
            
            # 清理 content - 更激进的清理策略
            # 1. 移除引用块（> 开头的行，包括搜索命令和引用链接）
            content = re.sub(r'^>.*?$', '', content, flags=re.MULTILINE)
            
            # 2. 移除所有 markdown 链接引用（**[文本](url)** · *domain* 格式）
            content = re.sub(r'\*\*\[.*?\]\(.*?\)\*\*\s*·\s*\*.*?\*', '', content)
            
            # 3. 移除普通 markdown 链接
            content = re.sub(r'\[.*?\]\(.*?\)', '', content)
            
            # 4. 移除代码块标记
            content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
            
            # 5. 移除多余的空行
            content = re.sub(r'\n{2,}', '\n', content).strip()
            
            print(f"📄 After cleaning (first 300 chars):")
            print(f"{content[:300]}")
            print(f"...")
            
            # 提取 JSON 数组 - 更智能的提取
            # 先尝试找到所有可能的 JSON 数组
            json_content = None
            
            # 方法1: 使用正则表达式查找简单的 JSON 数组模式
            # 匹配 [ ... ] 但允许内部有逗号分隔的对象
            simple_pattern = r'\[\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}(?:\s*,\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})*\s*\]'
            match = re.search(simple_pattern, content, re.DOTALL)
            
            if match:
                try:
                    test_json = match.group(0)
                    # 清理可能的干扰字符
                    test_json = re.sub(r'\n\s*\n', '\n', test_json)
                    # 验证是否为有效 JSON
                    test_obj = json.loads(test_json)
                    if isinstance(test_obj, list) and len(test_obj) > 0:
                        json_content = test_json
                        print(f"✅ Found valid JSON array (pattern matching): {len(json_content)} chars")
                except Exception as e:
                    print(f"⚠️ Pattern match found JSON-like structure but parsing failed: {e}")
            
            # 方法2: 如果方法1失败，使用括号匹配
            if not json_content:
                first_bracket = content.find('[')
                if first_bracket == -1:
                    print(f"⚠️ No JSON array found after cleaning")
                    print(f"📄 Cleaned content (first 500 chars): {content[:500]}")
                    return "NO_VERIFIED_TRENDS_FOUND"
                
                print(f"📍 Found '[' at position {first_bracket}, trying bracket matching...")
                
                # 从第一个 '[' 开始，找到匹配的 ']'
                bracket_count = 0
                last_bracket = -1
                in_string = False
                escape_next = False
                
                for i in range(first_bracket, len(content)):
                    char = content[i]
                    
                    # 处理字符串内的引号和转义
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"':
                        in_string = not in_string
                        continue
                    
                    # 只在字符串外计数括号
                    if not in_string:
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                last_bracket = i
                                break
                
                if last_bracket != -1:
                    json_content = content[first_bracket:last_bracket + 1].strip()
                    print(f"✅ Found JSON array (bracket matching): {len(json_content)} chars")
                else:
                    print(f"⚠️ Could not find matching ']' for JSON array")
            
            if not json_content:
                print(f"⚠️ Could not extract valid JSON array")
                print(f"📄 Content sample: {content[:500]}")
                return "NO_VERIFIED_TRENDS_FOUND"
            
            # 🔍 调试：打印提取后的 JSON 字符串
            print(f"🔍 Extracted JSON array length: {len(json_content)} chars")
            print(f"🔍 Extracted JSON (first 400 chars):")
            print(f"{json_content[:400]}")
            if len(json_content) > 400:
                print(f"...")
                print(f"🔍 Last 200 chars:")
                print(f"...{json_content[-200:]}")

            # -------------------------
            # 3. 解析 JSON
            # -------------------------
            try:
                topics = json.loads(json_content)
                print(f"✅ JSON parsed successfully: {len(topics)} topics found")
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON Parse Error: {e}")
                print(f"⚠️ Error position: line {e.lineno}, column {e.colno}")
                # 尝试修复常见的 JSON 问题
                try:
                    content_fixed = json_content
                    
                    # 1. 移除 trailing commas
                    content_fixed = re.sub(r',\s*]', ']', content_fixed)
                    content_fixed = re.sub(r',\s*}', '}', content_fixed)
                    
                    # 2. 移除可能的 BOM 和特殊字符
                    content_fixed = content_fixed.encode('utf-8').decode('utf-8-sig').strip()
                    
                    # 3. 移除所有控制字符（除了换行、回车、制表符）
                    content_fixed = ''.join(
                        char for char in content_fixed 
                        if ord(char) >= 32 or char in '\n\r\t'
                    )
                    
                    # 4. 移除嵌套的 JSON 数组（可能是重复内容）
                    # 如果发现字符串中包含 "[" 说明可能有嵌套
                    lines = content_fixed.split('\n')
                    cleaned_lines = []
                    in_string = False
                    for line in lines:
                        # 简单检测：如果行开头就是 '['，可能是重复的数组开始
                        if line.strip().startswith('[') and cleaned_lines:
                            print(f"🔧 Detected nested array start, truncating...")
                            break
                        cleaned_lines.append(line)
                    content_fixed = '\n'.join(cleaned_lines)
                    
                    print(f"🔧 Attempting to fix JSON...")
                    print(f"🔍 Fixed content (first 300 chars): {content_fixed[:300]}")
                    
                    topics = json.loads(content_fixed)
                    print(f"✅ JSON fixed and parsed successfully!")
                except Exception as e2:
                    print(f"⚠️ Retry Parse Error: {e2}")
                    print(f"❌ Failed content (first 500 chars): {repr(json_content[:500])}")
                    if len(json_content) > 500:
                        print(f"❌ Failed content (last 200 chars): {repr(json_content[-200:])}")
                    return "NO_VERIFIED_TRENDS_FOUND"

            # -------------------------
            # 4. 格式校验 + 示例数据检测 + URL提取
            # -------------------------
            verified = []
            if isinstance(topics, list):
                for t in topics:
                    # 检查必需字段（新格式：topic, platform, title, url）
                    if (
                        isinstance(t, dict)
                        and t.get("topic")
                        and t.get("platform")
                    ):
                        # 检测是否返回了示例数据（而非真实搜索结果）
                        topic_text = t.get("topic", "").lower()
                        
                        # 拒绝包含示例/模板关键词的数据
                        if any(x in topic_text for x in ["简要", "热点描述", "example", "示例"]):
                            print(f"⚠️ Detected template/example data, rejecting: {t.get('topic')}")
                            continue
                        
                        # 新增：拒绝泛泛的"热榜"、"热搜榜"等聚合页面（但要更精确）
                        if any(x in topic_text for x in ["今日热榜", "实时热搜榜", "热门排行榜", "综合热榜"]):
                            print(f"⚠️ Detected generic hot list, rejecting: {t.get('topic')}")
                            continue
                        
                        # 改进的验证逻辑：优先级 search_keyword > URL
                        has_search_keyword = bool(t.get("search_keyword"))
                        has_url_and_title = bool(t.get("url") and t.get("title"))
                        
                        # 如果有 search_keyword，基本上就接受（可以生成搜索链接）
                        if has_search_keyword:
                            keyword_text = t.get("search_keyword", "").lower()
                            # 只拒绝明显的模板关键词
                            if any(x in keyword_text for x in ["关键词示例", "keyword example", "搜索示例"]):
                                print(f"⚠️ Detected template keyword, rejecting: {t.get('search_keyword')}")
                                continue
                            print(f"✅ Valid topic with search_keyword: {t.get('topic')[:50]}...")
                            verified.append(t)
                            continue
                        
                        # 如果没有 search_keyword，检查 URL
                        if has_url_and_title:
                            url = t.get("url", "")
                            # 验证 URL 格式
                            if not (url.startswith("http://") or url.startswith("https://")):
                                print(f"⚠️ Invalid URL format, skipping: {url}")
                                continue
                            
                            # 检查是否是热榜聚合网站（这些要拒绝，因为不够具体）
                            hotlist_domains = [
                                "tophub.today",
                                "remenla.com",
                                "shenmehuole.com",
                                "imshuai.com",
                                "v2hot.com"
                            ]
                            
                            is_hotlist_site = any(domain in url for domain in hotlist_domains)
                            
                            if is_hotlist_site:
                                print(f"⚠️ Hotlist aggregator URL rejected: {url[:60]}...")
                                continue
                            
                            # URL 检查（已经优化过，更宽松）
                            if self._is_homepage_url(url):
                                print(f"⚠️ Homepage URL without search_keyword, skipping: {url[:60]}...")
                                continue
                            
                            print(f"✅ Valid topic with URL: {t.get('title')[:40]}... -> {url[:60]}...")
                            verified.append(t)
                            continue
                        
                        # 两者都没有，跳过
                        print(f"⚠️ Missing both URL and search_keyword, skipping topic")
                        continue

            if not verified:
                print("⚠️ No valid real topics found (only examples/templates)")
                
                # 重试机制
                if retry_count < max_retries:
                    print(f"🔄 Retrying... (attempt {retry_count + 1}/{max_retries})")
                    import time
                    time.sleep(2)  # 等待2秒
                    return self.search_hot_topics(user_profile, retry_count + 1, max_retries)
                
                # 降级策略：返回通用热点模板
                print("📋 Using fallback generic hot topics")
                return self._get_fallback_topics()

            print(f"🔍 VERIFIED HOT TOPICS ({current_time_str}):")
            for t in verified:
                if t.get('url') and t.get('title'):
                    # 新格式：带真实链接
                    print(f"- [{t['platform']}] {t['topic']}")
                    print(f"  📰 {t['title']}")
                    print(f"  🔗 {t['url'][:80]}...")
                elif t.get('search_keyword'):
                    # 旧格式：搜索关键词
                    print(f"- [{t['platform']}] {t['topic']} | 搜索词: {t['search_keyword']}")
                else:
                    print(f"- [{t['platform']}] {t['topic']}")

            return verified

        except Exception as e:
            print(f"⚠️ Search exception: {e}")
            import traceback
            traceback.print_exc()
            
            # 重试机制
            if retry_count < max_retries:
                print(f"🔄 Retrying after exception... (attempt {retry_count + 1}/{max_retries})")
                import time
                time.sleep(2)
                return self.search_hot_topics(user_profile, retry_count + 1, max_retries)
            
            # 降级策略：返回通用热点模板
            print("📋 Using fallback generic hot topics after exception")
            return self._get_fallback_topics()
    
    def _get_fallback_topics(self):
        """
        Fallback generic hot topics when web search fails
        返回通用的热点话题模板（不依赖实时搜索）
        """
        fallback_topics = [
            {
                "topic": "最近社交媒体上的热门话题讨论",
                "platform": "综合平台",
                "search_keyword": "热点话题 讨论",
                "title": "当前热门话题综合",
                "url": "https://tophub.today"
            },
            {
                "topic": "当下流行的网络热梗和文化现象",
                "platform": "虎扑/微博",
                "search_keyword": "网络热梗 文化",
                "title": "网络文化热点",
                "url": "https://www.zhihu.com/hot"
            }
        ]
        
        print("⚠️ Using 2 fallback topics (generic templates)")
        for i, t in enumerate(fallback_topics, 1):
            print(f"  {i}. [{t['platform']}] {t['topic']}")
        
        return fallback_topics
    


    def generate_idea_from_topics(self, user_profile, hot_topics):
        """Generate discussion post idea based on hot topics and user profile
        改进：生成的创意应该是"如何评论这个热点"，而不是独立的话题
        """
        # 构建热点列表
        topics_text = ""
        if isinstance(hot_topics, list):
            for i, t in enumerate(hot_topics):
                topics_text += f"{i+1}. [{t.get('platform', '')}] {t.get('topic', '')}\n"
        
        prompt = f"""
        You are a creative content strategist. Based on the user profile and current hot topics,
        generate 2-3 discussion post ideas that comment on or discuss these hot topics.
        
        User Profile:
        {user_profile}
        
        Hot Topics:
        {topics_text}
        
        Requirements:
        1. **Each idea should be about commenting on/discussing one of the hot topics above**
        2. One idea should be emotional/casual style
        3. One idea should be controversial or direct (a bit aggressive)
        4. One idea should be analytical/rational style
        5. **Must specify which hot topic to discuss**
        
        Return as JSON array:
        [
          {{
            "idea": "How to comment on this hot topic (e.g., analyze from fan perspective)",
            "hot_topic_index": 1,
            "angle": "Unique perspective or approach",
            "tone": "Casual/Aggressive/Analytical"
          }}
        ]
        """

        try:
            resp = requests.post(
                f"{self.search_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.search_api_key}"},
                json={
                    "model": self.search_model,
                    "web_search_options": {},
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                }
            )
            
            # Check response status
            if resp.status_code != 200:
                print(f"⚠️ API Error {resp.status_code}: {resp.text[:200]}")
                raise Exception(f"API returned status code {resp.status_code}")
            
            resp_data = resp.json()
            
            # Debug: Print response structure if choices not found
            if "choices" not in resp_data:
                print(f"⚠️ Unexpected response structure:")
                print(json.dumps(resp_data, ensure_ascii=False, indent=2)[:500])
                raise Exception("Response missing 'choices' field")
            
            ideas_text = resp_data["choices"][0]["message"]["content"].strip()
            
            # Try to extract JSON from response
            if "```json" in ideas_text:
                ideas_text = ideas_text.split("```json")[1].split("```")[0].strip()
            elif "```" in ideas_text:
                ideas_text = ideas_text.split("```")[1].split("```")[0].strip()
                
            ideas = json.loads(ideas_text)
            print(f"💡 Generated {len(ideas)} post ideas")
            return ideas
            
        except Exception as e:
            print(f"⚠️ Idea generation error: {e}")
            import traceback
            traceback.print_exc()
            return [{
                "idea": "对第一个热点发表个人看法",
                "hot_topic_index": 1,
                "angle": "个人观点分享",
                "tone": "Casual"
            }]

    def generate_text(self, user_profile, hot_topics, ideas, user_profile_path=None, profile_data=None):
        """Generate discussion post text based on REAL EXAMPLES (样例优先)
        核心逻辑：基于样例风格，对找到的热点事件进行评论/讨论
        
        Args:
            user_profile: User profile text
            hot_topics: Hot topics list
            ideas: Ideas list
            user_profile_path: Path to user profile (for RAG)
            profile_data: Profile data dict (for RAG)
        """
        
        # 加载真实样例 (支持RAG模式)
        examples = self.load_examples(user_profile_path=user_profile_path, profile_data=profile_data)
        
        if not examples:
            print("⚠️ No examples found, using fallback style templates")
            return self._generate_text_fallback(user_profile, hot_topics, ideas)
        
        print(f"✨ Generating text based on {len(examples)} real example(s)")
        
        # 构建样例文本，突出显示标题（特别是"理性讨论"等开头）
        examples_text_parts = []
        for i, ex in enumerate(examples):
            if ex.get('title'):
                examples_text_parts.append(
                    f"【样例 {i+1}】\n标题：{ex['title']}\n\n{ex['content']}"
                )
            else:
                examples_text_parts.append(
                    f"【样例 {i+1}】\n{ex['content']}"
                )
        
        examples_text = "\n\n---\n\n".join(examples_text_parts)
        
        # 构建热点上下文：提取具体的热点事件描述
        hot_topics_context = ""
        if isinstance(hot_topics, list) and len(hot_topics) > 0:
            hot_topics_context = "**当前可选的热点事件：**\n"
            for i, topic in enumerate(hot_topics):
                topic_desc = topic.get("topic", "")
                platform = topic.get("platform", "")
                hot_topics_context += f"{i+1}. [{platform}] {topic_desc}\n"
        
        # 新的 prompt：样例风格 + 热点评论
        prompt = f"""你是一个虎扑老用户，需要创作一篇讨论帖。

**核心任务：选择一个热点事件，用样例风格对其进行评论/讨论**

{examples_text}

---

{hot_topics_context}

**用户背景：**
{user_profile[:300]}

**创作要求：**
1. **从上面的热点中选择一个与用户兴趣相关的事件**
2. **完全模仿样例的风格、语气、用词、结构来评论这个热点**
3. 字数控制在200-400字以内
4. 直接、简洁，每句话都有信息量
5. 如果样例是"理性讨论"风格，你也用理性讨论开头
6. **必须在帖子中明确提到你选择的热点事件（如人物名、事件名）**
7. **如果有标题，直接输出标题内容，不要加"标题："等前缀**

**关键原则：**
- 样例怎么写，你就怎么写
- 不要泛泛而谈，要针对具体热点发表看法
- 保持样例的简洁和直接
- 如果第一行是标题，直接写标题内容即可，不需要"标题："前缀

请开始创作（只输出帖子正文）：
"""

        try:
            resp = requests.post(
                f"{self.search_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.search_api_key}"},
                json={
                    "model": self.search_model,
                    "web_search_options": {},
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,  # 降低温度，更贴近样例风格
                    "top_p": 0.9,
                }
            )
            
            if resp.status_code != 200:
                print(f"⚠️ Text Generation API Error {resp.status_code}: {resp.text[:200]}")
                raise Exception(f"API returned status code {resp.status_code}")
            
            resp_data = resp.json()
            
            if "choices" not in resp_data:
                print(f"⚠️ Unexpected response structure:")
                print(json.dumps(resp_data, ensure_ascii=False, indent=2)[:500])
                raise Exception("Response missing 'choices' field")
            
            full_content = resp_data["choices"][0]["message"]["content"].strip()
            
            # 清理思考标记和搜索元数据
            full_content = re.sub(r'```\s*\n?\s*\{[^}]*?"search_query".*?\}\s*```', '', full_content, flags=re.DOTALL)
            full_content = re.sub(r'```json\s*\n?\s*\{[^}]*?"search_query".*?\}\s*```', '', full_content, flags=re.DOTALL)
            full_content = re.sub(r'^>.*?$', '', full_content, flags=re.MULTILINE)
            full_content = re.sub(r'\*Thought for \d+s\*', '', full_content)
            full_content = re.sub(r'> \*\*.*?\*\*\n?', '', full_content, flags=re.MULTILINE)
            full_content = re.sub(r'\n{3,}', '\n\n', full_content)
            full_content = full_content.strip()
            
            # 字数检查和截断
            full_content = self._ensure_concise_text(full_content)
            
            print(f"✅ Generated {len(full_content)} chars based on examples")
            
            # 返回文本和一个简单的风格标记
            return full_content, [], {"name": "样例学习风格", "desc": "基于真实样例学习的风格"}
            
        except Exception as e:
            print(f"⚠️ Text generation error: {e}")
            import traceback
            traceback.print_exc()
            # 降级到备用方法
            return self._generate_text_fallback(user_profile, hot_topics, ideas)
    
    def _ensure_concise_text(self, text, max_chars=450):
        """
        Ensure text is concise and within character limit
        确保文本简短，不超过最大字符数
        """
        if len(text) <= max_chars:
            return text
        
        print(f"⚠️ Text too long ({len(text)} chars), truncating to {max_chars} chars")
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        
        # 保留前几段，直到接近限制
        result_paragraphs = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # 如果加上这段会超过限制
            if current_length + len(para) + 4 > max_chars:  # +4 for \n\n
                # 如果已经有至少1段了，就停止
                if result_paragraphs:
                    break
                # 如果还没有任何段落，截断这一段
                remaining = max_chars - current_length - 20  # 留空间加结尾
                if remaining > 50:
                    para = para[:remaining] + "..."
                    result_paragraphs.append(para)
                break
            
            result_paragraphs.append(para)
            current_length += len(para) + 2  # +2 for \n\n
        
        result = '\n\n'.join(result_paragraphs)
        
        # 如果结尾不是互动型结尾，添加一个
        if result and not any(end in result[-20:] for end in ['？', '?', '🤔', '🤝', 'JRs']):
            result += "\n\nJRs怎么看？🤔"
        
        print(f"✅ Truncated to {len(result)} chars")
        return result
    
    def _generate_text_fallback(self, user_profile, hot_topics, ideas):
        """Fallback method using style templates (原来的方法)"""
        
        # 定义三种截然不同的风格模板
        style_templates = {
            "casual": {
                "name": "虎扑乐子人 (吃瓜/玩梗)",
                "desc": "虎扑典型吃瓜网友，爱唠嗑、玩专属梗，自带吐槽buff，说话接地气又有梗",
                "weight": 3,
                "instructions": """
                1. **语气**：轻松调侃、自带戏谑感，像和JRs蹲在步行街主干道唠嗑，不端着
                2. **用词**：优先用虎扑热梗（蚌埠住了、绷不住了、谁懂啊），少用「绝绝子」这类软萌词
                3. **称呼**：多用虎扑专属称呼「JRs」「家人们」
                4. **Emoji**：用虎扑高频表情（🤣🤡🙉🔥😜），不堆砌
                5. **特点**：逻辑随缘，主打情绪共鸣+玩梗
                6. **字数**：200-400字，简短有力
                """
            },
            "aggressive": {
                "name": "虎扑暴躁老哥 (直球/怼人)",
                "desc": "虎扑硬核老坛友，看不惯就怼，说话不绕弯，吐槽直击痛点",
                "weight": 3,
                "instructions": """
                1. **语气**：冲、直球、带刺，不磨叽
                2. **用词**：虎扑式吐槽词（搁这扯犊子呢、纯纯nt、别洗了）
                3. **Emoji**：极少用，顶多结尾加😅/🙄
                4. **特点**：直击问题核心
                5. **字数**：200-350字，短句为主
                """
            },
            "analytical": {
                "name": "虎扑懂哥 (理智/摆事实)",
                "desc": "虎扑资深坛友，主打「摆数据、讲事实」",
                "weight": 4,
                "instructions": """
                1. **开头**：可以用"理性讨论"开头
                2. **语气**：冷静客观、不卑不亢
                3. **用词**：直接、简洁、有逻辑（先说缺点再说优点）
                4. **Emoji**：基本不用或极少用
                5. **特点**：先给结论再拆论据，有具体数据支撑
                6. **字数**：200-400字
                """
            }
        }

        styles = list(style_templates.keys())
        weights = [style_templates[s]["weight"] for s in styles]
        selected_style_key = random.choices(styles, weights=weights, k=1)[0]
        selected_style = style_templates[selected_style_key]
        
        print(f"🎭 Using fallback style: {selected_style['name']}")

        prompt = f"""你是一个虎扑用户，需要创作一篇讨论帖。

风格：{selected_style['name']}

参考信息：
- 用户画像：{user_profile}
- 热点话题：{hot_topics}
- 创意方向：{json.dumps(ideas, ensure_ascii=False)}

风格要求：
{selected_style['instructions']}

核心原则：
- 字数200-400字
- 直接、简洁、不废话
- 每句话都有信息量
- 如果有标题，直接写标题内容，不要加"标题："等前缀

请开始创作（只输出帖子正文）：
"""

        try:
            resp = requests.post(
                f"{self.search_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.search_api_key}"},
                json={
                    "model": self.search_model,
                    "web_search_options": {},
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            
            if resp.status_code != 200:
                raise Exception(f"API error {resp.status_code}")
            
            resp_data = resp.json()
            full_content = resp_data["choices"][0]["message"]["content"].strip()
            
            # 清理
            full_content = re.sub(r'^>.*?$', '', full_content, flags=re.MULTILINE)
            full_content = re.sub(r'\n{3,}', '\n\n', full_content)
            full_content = full_content.strip()
            
            # 字数检查和截断
            full_content = self._ensure_concise_text(full_content)
            
            print(f"✅ Generated {len(full_content)} chars (fallback)")
            
        except Exception as e:
            print(f"⚠️ Fallback generation error: {e}")
            full_content = f"这是一篇关于热点话题的讨论帖。{selected_style['name']}风格的内容生成失败。"
        
        return full_content, [], selected_style
    
    def _is_homepage_url(self, url):
        """检测是否为首页链接（需要过滤掉）
        优化：更宽松的检查，减少误判
        """
        if not url or url == "#":
            return True
        
        # 提取路径部分
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        query = parsed.query
        
        # 宽松策略：如果有路径（即使很短），就认为不是首页
        # 例如：/cba/ 或 /news/ 等都应该保留
        if path:
            return False
        
        # 如果有查询参数（即使只是一个问号），也保留
        # 因为某些动态网站的URL就是这样的
        # 例如：https://sports.sina.cn/cba/? 可能是有效的分类页面
        
        # 只有在完全没有路径和查询参数时才判断为首页
        if not path and not query:
            return True
        
        return False

    def _generate_search_url(self, platform, keyword):
        """Generate real search URL based on platform and keyword
        修改：使用精确的关键词生成搜索链接
        """
        # 清理关键词中可能的平台名称
        kw_cleaned = keyword
        platform_names = ["微博", "知乎", "虎扑", "b站", "bilibili", "抖音", "小红书"]
        for pname in platform_names:
            kw_cleaned = kw_cleaned.replace(pname, "").strip()
        
        kw_encoded = urllib.parse.quote(kw_cleaned)
        p = platform.lower()
        
        if "b站" in p or "bilibili" in p:
            return f"https://search.bilibili.com/all?keyword={kw_encoded}"
        elif "小红书" in p:
            return f"https://www.xiaohongshu.com/search_result?keyword={kw_encoded}"
        elif "知乎" in p:
            return f"https://www.zhihu.com/search?type=content&q={kw_encoded}"
        elif "抖音" in p:
            return f"https://www.douyin.com/search/{kw_encoded}"
        elif "微博" in p:
            return f"https://s.weibo.com/weibo?q={kw_encoded}"
        elif "虎扑" in p or "hupu" in p:
            return f"https://s.hupu.com/all?q={kw_encoded}"
        else:
            return f"https://www.baidu.com/s?wd={kw_encoded}"

    def generate_images(self, user_profile, text_content, ideas, output_dir):
        """Generate images for the post"""
        word_count = len(text_content.strip())
        num_images = 1 if word_count <= 400 else (2 if word_count <= 900 else 3)
        print(f"Text length: {word_count} chars, generating {num_images} images")

        idea_types = [idea.get("angle", "") for idea in ideas]
        idea_prompt = f"Content angles: {', '.join(idea_types)}"

        image_paths = []
        max_retries = 3

        for i in range(num_images):
            image_path = os.path.join(output_dir, f"discussion_post_{i+1}.png")
            success = False
            for attempt in range(1, max_retries + 1):
                prompt = f"""
                Generate an image for a casual discussion post (image {i+1}/{num_images}):
                User Profile: {user_profile}
                Post Content: {text_content[:500]}...
                {idea_prompt}
                Requirements: Natural, relatable image suitable for social media discussion posts. 
                Can be casual, humorous, or reflective. No text in image.
                """
                try:
                    resp = requests.post(
                        f"{self.generate_base_url}/models/gemini-2.5-flash-image-preview:generateContent",
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.generate_api_key}"},
                        json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.7}},
                        timeout=60
                    )
                    if resp.status_code == 200:
                        parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        for part in parts:
                            if "inlineData" in part:
                                with open(image_path, "wb") as f:
                                    f.write(base64.b64decode(part["inlineData"]["data"]))
                                image_paths.append(image_path)
                                success = True
                                print(f"✅ Image {i+1} saved")
                                break
                    if success: break
                except Exception as e:
                    print(f"⚠️ Image generation error: {e}")
            if not success: print(f"❌ Image {i+1} skipped")
        return image_paths

    def _render_markdown(self, text):
        """
        Simple Markdown to HTML converter
        Converts **text** to bold styling
        NOTE: Links are handled separately as cards, not inline
        """
        # Remove any markdown links [text](url) that might have slipped through cleaning
        # We don't want inline links, only card-style links
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Handle bold **text** -> <strong>text</strong>
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #333; font-weight: 700;">\1</strong>', text)
        
        # Handle headers (in case AI outputs them)
        text = re.sub(r'^###\s+(.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        
        # Handle list items
        text = re.sub(r'^\-\s+(.*?)$', r'• \1', text, flags=re.MULTILINE)
        text = re.sub(r'^\•\s+(.*?)$', r'• \1', text, flags=re.MULTILINE)

        return text

    # --- FIX: 增加了 selected_style 参数 ---
    def generate_html_post(self, text_content, image_paths, links, hot_topics_summary, selected_style, output_path="discussion_post.html"):
        """Generate HTML discussion post with Markdown rendering"""
        
        # Remove "标题：" prefix if exists at the beginning
        text_content = re.sub(r'^标题：\s*', '', text_content.strip())
        text_content = re.sub(r'^Title:\s*', '', text_content.strip(), flags=re.IGNORECASE)
        
        # Split paragraphs by double newlines
        raw_paragraphs = [p for p in text_content.split('\n\n') if p.strip()]
        
        # Render Markdown for each paragraph
        processed_paragraphs = []
        for i, p in enumerate(raw_paragraphs):
            # First paragraph is likely the title - make it bold and larger
            if i == 0 and len(p) < 200:  # Titles are usually shorter
                rendered_p = f'<h2 style="font-size: 1.5em; font-weight: 700; color: #1a1a1a; margin-bottom: 0.8em; line-height: 1.4;">{self._render_markdown(p)}</h2>'
            else:
                rendered_p = self._render_markdown(p)
            processed_paragraphs.append(rendered_p)
            
        paragraphs = processed_paragraphs

        insertions = []
        
        # Insert images
        for i, img_path in enumerate(image_paths):
            insertions.append({"type": "image", "content": img_path, "index": i})
        # Insert links
        for link in links:
            insertions.append({"type": "link", "content": link})
            
        html_parts = []
        num_paras = len(paragraphs)
        
        if num_paras == 0:
            html_parts.append(text_content)
        else:
            num_inserts = len(insertions)
            if num_inserts > 0:
                step = max(1, num_paras // (num_inserts + 1))
                current_insert_idx = 0
                for i, para in enumerate(paragraphs):
                    if para.startswith('<h2') or para.startswith('<h3'):
                         html_parts.append(para)
                    else:
                         html_parts.append(f"<p>{para}</p>")
                         
                    if current_insert_idx < num_inserts:
                        if (i + 1) % step == 0 or i == num_paras - 1:
                            item = insertions[current_insert_idx]
                            if item["type"] == "image":
                                html_parts.append(self._create_image_tag(item["content"], item["index"]))
                            elif item["type"] == "link":
                                html_parts.append(self._create_link_tag(item["content"]))
                            current_insert_idx += 1
                            if i == num_paras - 1:
                                while current_insert_idx < num_inserts:
                                    item = insertions[current_insert_idx]
                                    if item["type"] == "image":
                                        html_parts.append(self._create_image_tag(item["content"], item["index"]))
                                    elif item["type"] == "link":
                                        html_parts.append(self._create_link_tag(item["content"]))
                                    current_insert_idx += 1
            else:
                for para in paragraphs:
                    if para.startswith('<h2') or para.startswith('<h3'):
                        html_parts.append(para)
                    else:
                        html_parts.append(f"<p>{para}</p>")

        html_content = "\n".join(html_parts)
                
        html_template = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>热点讨论帖子</title>
            <style>
                body {{ font-family: 'Helvetica Neue', Helvetica, 'Microsoft YaHei', sans-serif; line-height: 1.8; max-width: 850px; margin: 0 auto; padding: 15px; background: #f8f9fa; color: #333; }}
                .post-container {{ background: white; border-radius: 10px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 10px 0; }}
                
                /* Header with hot topic badge */
                .post-header {{ display: flex; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #f0f0f0; }} 
                .avatar {{ width: 48px; height: 48px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin-right: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }} 
                .user-info {{ flex: 1; }}
                .user-info h3 {{ margin: 0; font-size: 18px; font-weight: 600; }} 
                .post-time {{ color: #999; font-size: 13px; margin-top: 4px; }} 
                .hot-badge {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 10px; display: inline-block; }}
                
                /* Content styling - casual and readable */
                .post-content {{ font-size: 16px; color: #2c3e50; letter-spacing: 0.02em; line-height: 1.9; }}
                .post-content p {{ margin: 1em 0; text-align: left; }}
                .post-content strong {{ color: #000; font-weight: 700; background: linear-gradient(to bottom, transparent 60%, #fff3cd 60%); }}
                .post-content h3 {{ font-size: 1.2em; margin-top: 1.5em; margin-bottom: 0.5em; color: #1a1a1a; }}
                
                .post-image {{ margin: 20px -25px; width: calc(100% + 50px); text-align: center; }}
                .post-image img {{ width: 100%; display: block; border-radius: 8px; }}
                .image-caption {{ color: #999; font-size: 13px; margin-top: 8px; font-style: italic; padding: 0 25px; }}

                /* Link card styling */
                .link-card {{
                    display: flex;
                    align-items: center;
                    background: #fcfcfc;
                    border: 1px solid #eee;
                    padding: 12px 15px;
                    margin: 25px 0;
                    text-decoration: none;
                    border-radius: 8px;
                    transition: all 0.2s;
                }}
                .link-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
                .link-card.platform-b站 {{ border-left: 5px solid #23ade5; }}
                .link-card.platform-小红书 {{ border-left: 5px solid #ff2442; }}
                .link-card.platform-知乎 {{ border-left: 5px solid #0084ff; }}
                .link-card.platform-抖音 {{ border-left: 5px solid #1c1e21; }}
                .link-card.platform-微博 {{ border-left: 5px solid #ea5d5c; }}
                .link-card.platform-虎扑 {{ border-left: 5px solid #ff6600; }}
                .link-card.platform-腾讯新闻 {{ border-left: 5px solid #00a4ff; }}
                .link-card.platform-网易新闻 {{ border-left: 5px solid #c00000; }}
                .link-card.platform-澎湃新闻 {{ border-left: 5px solid #2b7fd1; }}
                
                .link-info {{ flex: 1; }}
                .link-platform-tag {{ 
                    font-size: 12px; font-weight: bold; margin-bottom: 4px; display: inline-block; padding: 2px 6px; border-radius: 4px; color: white;
                }}
                .tag-b站 {{ background: #23ade5; }}
                .tag-小红书 {{ background: #ff2442; }}
                .tag-知乎 {{ background: #0084ff; }}
                .tag-抖音 {{ background: #000; }}
                .tag-微博 {{ background: #ea5d5c; }}
                .tag-虎扑 {{ background: #ff6600; }}
                .tag-腾讯新闻 {{ background: #00a4ff; }}
                .tag-网易新闻 {{ background: #c00000; }}
                .tag-澎湃新闻 {{ background: #2b7fd1; }}
                
                .link-title {{ font-weight: bold; color: #333; font-size: 15px; margin-top: 2px; }}
                .link-action {{ color: #999; font-size: 12px; margin-top: 4px; }}
                .link-icon {{ font-size: 24px; margin-right: 15px; }}

                .footer {{ margin-top:30px; border-top:1px solid #eee; padding-top:15px; color:#ccc; font-size:12px; text-align:center; }}
                .hot-topic-tag {{ display: inline-block; background: #fff3cd; color: #856404; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin: 5px 5px 5px 0; }}

                @media (max-width: 600px) {{ 
                    .post-container {{ padding: 15px; }} 
                    .post-image {{ margin: 15px -15px; width: calc(100% + 30px); }}
                }}
            </style>
        </head>
        <body>
            <div class="post-container">
                <div class="post-header">
                    <div class="avatar"></div>
                    <div class="user-info">
                        <h3>论坛活跃JR <span class="hot-badge">🔥 {selected_style['name'].split()[0]}</span></h3>
                        <div class="post-time">{datetime.now().strftime('%Y年%m月%d日 %H:%M')}</div>
                    </div>
                </div>
                <div class="post-content">{html_content}</div>
                <div class="footer">
                    <div style="margin-bottom: 10px;">
                        {' '.join([f'<span class="hot-topic-tag">#{topic}</span>' for topic in ['热点话题', '讨论', 'AI生成']])}
                    </div>
                    Generated by AI • {len(image_paths)} Images • {len(links)} Links • Based on Hot Topics
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        return output_path

    def _create_image_tag(self, image_path, index):
        return f'<div class="post-image"><img src="{os.path.basename(image_path)}"><div class="image-caption">图 {index + 1}</div></div>'

    def _create_link_tag(self, link_data):
        title = link_data.get('title', '相关内容')
        platform = link_data.get('platform', '网页').strip()
        url = link_data.get('url', '#')
        
        css_class = "platform-other"
        tag_class = "tag-other"
        icon = "🔗"
        
        p = platform.lower()
        if "b站" in p or "bilibili" in p:
            css_class = "platform-b站"
            tag_class = "tag-b站"
            icon = "📺"
        elif "小红书" in p:
            css_class = "platform-小红书"
            tag_class = "tag-小红书"
            icon = "📕"
        elif "知乎" in p:
            css_class = "platform-知乎"
            tag_class = "tag-知乎"
            icon = "❓"
        elif "抖音" in p:
            css_class = "platform-抖音"
            tag_class = "tag-抖音"
            icon = "🎵"
        elif "微博" in p or "weibo" in p:
            css_class = "platform-微博"
            tag_class = "tag-微博"
            icon = "👁️"
        elif "虎扑" in p or "hupu" in p:
            css_class = "platform-虎扑"
            tag_class = "tag-虎扑"
            icon = "🏀"
        elif "腾讯" in p or "qq" in p or "tencent" in p:
            css_class = "platform-腾讯新闻"
            tag_class = "tag-腾讯新闻"
            icon = "📰"
        elif "网易" in p or "netease" in p or "163" in p:
            css_class = "platform-网易新闻"
            tag_class = "tag-网易新闻"
            icon = "📰"
        elif "澎湃" in p or "thepaper" in p:
            css_class = "platform-澎湃新闻"
            tag_class = "tag-澎湃新闻"
            icon = "📰"
            
        return f'''
        <a href="{url}" class="link-card {css_class}" target="_blank">
            <div class="link-icon">{icon}</div>
            <div class="link-info">
                <span class="link-platform-tag {tag_class}">{platform}</span>
                <div class="link-title">{title}</div>
                <div class="link-action">点击去 {platform} 查看详情 &gt;</div>
            </div>
        </a>
        '''

    def _filter_relevant_links(self, text_content, hot_topics):
        """
        Filter hot topics to keep only those relevant to the post content
        使用 LLM 判断哪些热点与帖子内容真正相关（宽松模式）
        """
        if not isinstance(hot_topics, list) or len(hot_topics) == 0:
            return []
        
        # 如果只有1个热点，直接返回
        if len(hot_topics) == 1:
            print(f"   📌 Only 1 hot topic, keeping it")
            return hot_topics
        
        # 如果只有2个热点，也比较宽松（保留至少1个）
        if len(hot_topics) == 2:
            print(f"   📌 Only 2 hot topics, using relaxed filtering")
        
        print(f"   🔍 Filtering {len(hot_topics)} hot topics for relevance...")
        
        # 构建筛选 prompt
        topics_summary = []
        for i, t in enumerate(hot_topics):
            topics_summary.append(f"{i+1}. {t.get('topic', '')} (平台: {t.get('platform', '')})")
        
        filter_prompt = f"""你需要判断哪些热点话题与这篇帖子内容相关。

帖子内容（前800字）：
{text_content[:800]}

可选的热点话题：
{chr(10).join(topics_summary)}

要求：
1. 选择与帖子内容**相关或可能相关**的热点（标准宽松，只要有关联即可）
2. 帖子可能是基于这些热点写的，即使没有明确提到热点名称，也可能在讨论相关话题
3. 保留 1-2 个最相关的热点
4. **如果不确定，倾向于保留而非删除**（宁可多不可少）

返回JSON格式（只返回序号数组）：
{{"selected": [1]}}  # 至少保留1个
"""

        try:
            resp = requests.post(
                f"{self.search_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.search_api_key}"},
                json={
                    "model": self.search_model,
                    "messages": [{"role": "user", "content": filter_prompt}],
                    "temperature": 0.5,  # 提高温度，更宽松的判断
                },
                timeout=30
            )
            
            if resp.status_code != 200:
                print(f"   ⚠️ Filter API error, keeping first topic as fallback")
                return [hot_topics[0]] if hot_topics else []
            
            resp_data = resp.json()
            content = resp_data["choices"][0]["message"]["content"].strip()
            
            # 提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # 查找 JSON 对象
            json_match = re.search(r'\{[^}]*"selected"[^}]*\}', content)
            if json_match:
                content = json_match.group(0)
            
            result = json.loads(content)
            selected_indices = result.get("selected", [])
            
            if not selected_indices:
                # 宽松模式：如果没选中任何热点，默认保留第一个
                print(f"   ⚠️ No topics selected by filter, keeping first topic as fallback")
                return [hot_topics[0]] if hot_topics else []
            
            # 根据选中的序号筛选热点
            filtered = []
            for idx in selected_indices:
                if 1 <= idx <= len(hot_topics):
                    filtered.append(hot_topics[idx - 1])
                    print(f"   ✅ Kept topic {idx}: {hot_topics[idx - 1].get('topic', '')[:50]}...")
            
            # 如果筛选后没有结果，保留第一个作为降级
            if not filtered:
                print(f"   ⚠️ Filtering resulted in empty list, keeping first topic")
                return [hot_topics[0]] if hot_topics else []
            
            return filtered
            
        except Exception as e:
            print(f"   ⚠️ Filter exception ({e}), keeping first topic as fallback")
            # 降级策略：保留第一个热点而不是全部
            return [hot_topics[0]] if hot_topics else []
    
    def _extract_links_from_topics(self, topics):
        """
        Extract links from filtered topics
        从筛选后的热点中提取链接
        """
        links = []
        if not isinstance(topics, list):
            return links
            
        for topic in topics:
            if not isinstance(topic, dict):
                continue
                
            # 新格式：有 URL 和 title
            if topic.get("url") and topic.get("title"):
                url = topic.get("url", "")
                # 过滤首页链接
                if not self._is_homepage_url(url):
                    links.append({
                        "title": topic.get("title", "相关新闻"),
                        "platform": topic.get("platform", "网页"),
                        "url": url
                    })
                    print(f"   ✅ [新格式] {topic.get('title', '')[:50]}...")
            
            # 旧格式：只有 search_keyword，生成搜索链接
            elif topic.get("search_keyword") and topic.get("platform"):
                keyword = topic.get("search_keyword", "")
                platform = topic.get("platform", "")
                search_url = self._generate_search_url(platform, keyword)
                
                # 使用更清晰的标题格式
                topic_desc = topic.get("topic", "")
                title = f"{topic_desc}" if topic_desc else f"{platform}搜索: {keyword}"
                
                links.append({
                    "title": title[:80],  # 限制标题长度
                    "platform": platform,
                    "url": search_url
                })
                print(f"   ⚠️ [旧格式/搜索] {title[:50]}... | 关键词: {keyword}")
        
        return links
    
    def _generate_ideas_from_profile(self, user_profile):
        """
        Generate post ideas based on user profile (fallback when no hot topics)
        基于用户画像生成创意（无热点时的降级方案）
        """
        prompt = f"""Based on the user profile, generate 2-3 discussion post ideas for a Hupu-style forum.

User Profile:
{user_profile}

Requirements:
1. Generate ideas that match the user's interests and posting style
2. Ideas should be relatable topics that don't require real-time hot topics
3. Can be about general interests, opinions, or experiences
4. One casual/fun idea, one opinion-based idea, one analytical idea

Return as JSON array:
[
  {{
    "idea": "Discussion post idea",
    "angle": "Unique perspective or approach",
    "tone": "Casual/Opinionated/Analytical"
  }}
]
"""

        try:
            resp = requests.post(
                f"{self.search_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.search_api_key}"},
                json={
                    "model": self.search_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                },
                timeout=30
            )
            
            if resp.status_code != 200:
                raise Exception(f"API error {resp.status_code}")
            
            resp_data = resp.json()
            ideas_text = resp_data["choices"][0]["message"]["content"].strip()
            
            # Extract JSON
            if "```json" in ideas_text:
                ideas_text = ideas_text.split("```json")[1].split("```")[0].strip()
            elif "```" in ideas_text:
                ideas_text = ideas_text.split("```")[1].split("```")[0].strip()
                
            ideas = json.loads(ideas_text)
            print(f"💡 Generated {len(ideas)} profile-based ideas")
            return ideas
            
        except Exception as e:
            print(f"⚠️ Idea generation error: {e}")
            # Fallback ideas
            return [{
                "idea": "分享最近的观点和想法",
                "angle": "个人经验和见解",
                "tone": "Casual"
            }]
    
    def _generate_text_from_profile(self, user_profile, ideas, user_profile_path=None, profile_data=None):
        """
        Generate text based on user profile and examples (no hot topics)
        基于用户画像和样例生成内容（无热点）
        
        Args:
            user_profile: User profile text
            ideas: Ideas list
            user_profile_path: Path to user profile (for RAG)
            profile_data: Profile data dict (for RAG)
        """
        # 加载样例 (支持RAG模式)
        examples = self.load_examples(user_profile_path=user_profile_path, profile_data=profile_data)
        
        if not examples:
            print("⚠️ No examples, using basic fallback")
            return self._generate_text_basic_fallback(user_profile, ideas)
        
        print(f"✨ Generating profile-based text using {len(examples)} example(s)")
        
        # 构建样例文本
        examples_text_parts = []
        for i, ex in enumerate(examples):
            if ex.get('title'):
                examples_text_parts.append(
                    f"【样例 {i+1}】\n标题：{ex['title']}\n\n{ex['content']}"
                )
            else:
                examples_text_parts.append(
                    f"【样例 {i+1}】\n{ex['content']}"
                )
        
        examples_text = "\n\n---\n\n".join(examples_text_parts)
        
        prompt = f"""你是一个虎扑老用户，需要创作一篇讨论帖。

**核心任务：完全模仿以下真实样例的风格（权重90%）**

{examples_text}

---

**参考信息：**
- 你的背景：{user_profile}
- 话题方向：{json.dumps(ideas, ensure_ascii=False, indent=2)}

**创作要求：**
1. **完全模仿样例的风格、语气、用词**
2. 字数200-400字以内
3. 直接、简洁，不废话
4. 如果是理性分析类，可以用"理性讨论"开头
5. **如果有标题，直接输出标题内容，不要加"标题："等前缀**

**关键原则：样例怎么写，你就怎么写**

请开始创作（只输出帖子正文）：
"""

        try:
            resp = requests.post(
                f"{self.search_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.search_api_key}"},
                json={
                    "model": self.search_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                },
                timeout=60
            )
            
            if resp.status_code != 200:
                raise Exception(f"API error {resp.status_code}")
            
            resp_data = resp.json()
            full_content = resp_data["choices"][0]["message"]["content"].strip()
            
            # 清理
            full_content = re.sub(r'^>.*?$', '', full_content, flags=re.MULTILINE)
            full_content = re.sub(r'\n{3,}', '\n\n', full_content)
            full_content = full_content.strip()
            
            # 字数检查和截断
            full_content = self._ensure_concise_text(full_content)
            
            print(f"✅ Generated {len(full_content)} chars (profile-based)")
            
            return full_content, [], {"name": "用户画像风格", "desc": "基于用户画像和样例"}
            
        except Exception as e:
            print(f"⚠️ Profile-based generation error: {e}")
            return self._generate_text_basic_fallback(user_profile, ideas)
    
    def _generate_text_basic_fallback(self, user_profile, ideas):
        """Basic fallback when everything else fails"""
        text = f"""JRs好，想聊个事。

{ideas[0].get('idea', '最近的一些想法')}

大家怎么看？欢迎讨论🤝"""
        
        print(f"⚠️ Using basic fallback text ({len(text)} chars)")
        return text, [], {"name": "基础降级", "desc": "最简单的降级方案"}

    def __call__(self, user_profile_path, output_dir):
        """Main execution flow - 确保内容一定会生成"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load user profile
        with open(user_profile_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        
        user_profile = profile_data.get("profile_text", json.dumps(profile_data, ensure_ascii=False))
        
        print("\n🔍 Step 1: Searching for hot topics...")
        hot_topics = self.search_hot_topics(user_profile)
        
        # 新逻辑：即使搜索失败也继续，而不是直接退出
        use_fallback_mode = False
        if hot_topics == "NO_VERIFIED_TRENDS_FOUND":
            print("⚠️ Web search failed. Using profile-based content generation mode.")
            use_fallback_mode = True
            # 不再返回错误，而是继续生成

        # --- 保存热点信息 ---
        topics_save_path = os.path.join(output_dir, "topics.json")
        parsed_topics = []
        raw_text_content = ""

        try:
            if isinstance(hot_topics, list):
                parsed_topics = [t.get('topic', '未知话题') for t in hot_topics]
                raw_text_content = json.dumps(hot_topics, ensure_ascii=False, indent=2)
            else:
                parsed_topics = ["基于用户画像生成（无实时热点）"]
                raw_text_content = "Fallback mode: No hot topics available"

            with open(topics_save_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "mode": "fallback" if use_fallback_mode else "hot_topics",
                    "structured_topics": parsed_topics,
                    "raw_text": raw_text_content
                }, f, ensure_ascii=False, indent=2)
            print(f"💾 Topics info saved to: {topics_save_path}")
        except Exception as e:
            print(f"⚠️ Failed to save topics.json: {e}")
        
        # --- 生成创意和内容 ---
        if use_fallback_mode:
            # 降级模式：基于用户画像生成
            print("\n💡 Step 2: Generating post ideas from user profile (fallback mode)...")
            ideas = self._generate_ideas_from_profile(user_profile)
        else:
            # 正常模式：基于热点生成
            print("\n💡 Step 2: Generating post ideas from hot topics...")
            ideas = self.generate_idea_from_topics(user_profile, hot_topics)
        
        # Only use the first idea
        if len(ideas) > 1:
            print(f"💡 Generated {len(ideas)} ideas, using only the first one")
            ideas = [ideas[0]]
        else:
            print(f"💡 Generated {len(ideas)} idea")
        
        # Save ideas
        with open(os.path.join(output_dir, "discussion_ideas.json"), 'w', encoding='utf-8') as f:
            json.dump(ideas, f, ensure_ascii=False, indent=2)
        
        print("\n📝 Step 3: Generating discussion post text...")
        if use_fallback_mode:
            # 降级模式：基于用户画像和样例生成
            text, _, style = self._generate_text_from_profile(user_profile, ideas, 
                                                              user_profile_path=user_profile_path,
                                                              profile_data=profile_data)
        else:
            # 正常模式：基于热点生成
            text, _, style = self.generate_text(user_profile, hot_topics, ideas,
                                                user_profile_path=user_profile_path,
                                                profile_data=profile_data)
        
        # --- 处理链接 ---
        links = []
        if use_fallback_mode:
            # 降级模式：不添加链接（因为没有相关热点）
            print("\n🔗 Step 4: Skipping link generation (fallback mode, no hot topics)")
        else:
            # 正常模式：筛选和提取链接
            print("\n🎯 Step 4: Filtering relevant hot topics for the post...")
            relevant_topics = self._filter_relevant_links(text, hot_topics)
            
            print(f"\n🔗 Step 5: Extracting links from {len(relevant_topics)} relevant topic(s)...")
            links = self._extract_links_from_topics(relevant_topics)
        
        if links:
            print(f"   📊 Total {len(links)} links extracted")
        else:
            print(f"   ℹ️ No links added to this post")
        
        print("\n🖼️ Step 6: Generating images...")
        # images = self.generate_images(user_profile, text, ideas, output_dir)
        images = []  # ---暂且关闭image功能---

        print("\n🌐 Step 7: Generating HTML...")
        html_path = os.path.join(output_dir, "discussion_post_v0.html")
        
        # 调试信息
        print(f"   📝 Text: {len(text)} chars")
        print(f"   🖼️  Images: {len(images)}")
        print(f"   🔗 Links: {len(links)}")
        if links:
            for i, link in enumerate(links):
                print(f"      Link {i+1}: {link.get('title', '')[:40]}... [{link.get('platform', '')}]")
        
        self.generate_html_post(text, images, links, parsed_topics, style, html_path)
        print(f"✅ Complete: {html_path}")
        
        # ========================= Reflection Mechanism =========================
        # 评估链接-文本相关性，如果低于阈值则触发reflection
        current_html_path = html_path
        reflection_history = []
        removed_links_history = []  # 记录所有已删除的链接（避免重复添加）
        
        # 连续不变计数器（用于提前停止）
        no_improvement_count = 0
        last_best_score = None
        NO_IMPROVEMENT_THRESHOLD = 5  # 连续5-6次不变后停止（可配置）
        
        # 只在非fallback模式且有链接时才进行reflection
        if self.reflection_enabled and not use_fallback_mode and links:
            print("\n" + "="*80)
            print(f"🔄 启动Reflection机制（虎扑链接-文本评估，最多{self.max_reflection_iterations}次迭代）")
            print(f"   阈值: Link-Text GroupScore ≥ {self.reflection_threshold}")
            print(f"   优化: 缓存+并行化，连续{NO_IMPROVEMENT_THRESHOLD}次不变自动停止")
            print("="*80)
            
            for iteration in range(self.max_reflection_iterations):
                print(f"\n────────────────────────────────────────────────────────────────────────────────")
                print(f"📊 第{iteration+1}次评估 (当前版本: v{iteration})")
                print(f"────────────────────────────────────────────────────────────────────────────────")
                
                # 1. 计算Link-Text GroupScore
                print("\n1️⃣  计算Link-Text GroupScore...")
                try:
                    eval_result = evaluate_file_links(
                        html_path=current_html_path,
                        evaluator=self.text_evaluator,
                        verbose=False
                    )
                    groupscore = eval_result.group_score_mean  # 使用算术平均（更稳定）
                    print(f"   ✅ GroupScore (Mean): {groupscore:.4f}")
                    print(f"      (Harmonic: {eval_result.group_score_harmonic:.4f})")
                    print(f"      - 评估了 {eval_result.num_pairs} 个链接")
                except Exception as e:
                    print(f"   ❌ GroupScore计算失败: {e}")
                    groupscore = 0.0  # 失败则设为0，强制触发reflection
                
                reflection_history.append({
                    "version": f"v{iteration}",
                    "groupscore": groupscore,
                    "html_path": current_html_path
                })
                
                # 2. 判断是否达标
                if groupscore >= self.reflection_threshold:
                    print(f"\n   ✅ GroupScore {groupscore:.4f} 达到阈值 {self.reflection_threshold}，停止Reflection。")
                    break
                
                # 2.5. 检查连续不变（提前停止优化）
                if last_best_score is not None:
                    # 获取当前最佳分数
                    current_best = max(reflection_history, key=lambda x: x.get("groupscore", 0))
                    current_best_score = current_best["groupscore"]
                    
                    # 如果最佳分数没有提升（允许小的浮动，0.001）
                    if abs(current_best_score - last_best_score) < 0.001:
                        no_improvement_count += 1
                        print(f"\n   ℹ️  连续 {no_improvement_count} 次迭代最佳分数未提升 ({current_best_score:.4f})")
                        if no_improvement_count >= NO_IMPROVEMENT_THRESHOLD:
                            print(f"   ⏹️  连续 {NO_IMPROVEMENT_THRESHOLD} 次不变，提前停止Reflection")
                            break
                    else:
                        # 有提升，重置计数器
                        no_improvement_count = 0
                        last_best_score = current_best_score
                else:
                    # 第一次迭代，记录初始最佳分数
                    last_best_score = groupscore
                
                # 🎯 第2+次Reflection：检查是否需要回退到历史最佳版本
                if iteration >= 1 and len(reflection_history) >= 2:
                    # 找出历史最佳版本
                    best_record = max(reflection_history, key=lambda x: x.get("groupscore", 0))
                    best_score = best_record["groupscore"]
                    best_html_path = best_record["html_path"]
                    best_version = best_record["version"]
                    
                    print(f"\n💡 历史最佳版本检查:")
                    print(f"   - 历史最高分: {best_version} ({best_score:.4f})")
                    print(f"   - 当前分数: v{iteration} ({groupscore:.4f})")
                    
                    # 如果当前版本不是最佳版本，且差距明显，切换到最佳版本
                    if current_html_path != best_html_path and groupscore < best_score:
                        score_gap = best_score - groupscore
                        print(f"   🔄 分数差距 {score_gap:.4f}，切换到最佳版本进行优化")
                        current_html_path = best_html_path
                        groupscore = best_score
                        
                        # 更新reflection_history，标记这次切换
                        reflection_history[-1]['switched_to_best'] = True
                        
                        # 如果切换后已经达标，直接结束
                        if groupscore >= self.reflection_threshold:
                            print(f"   ✅ 最佳版本已达标，无需继续优化")
                            break
                    else:
                        print(f"   ✅ 当前版本已是最佳或接近最佳")
                
                # 3. 分析并改进
                print(f"\n3️⃣  分析链接相关性问题...")
                print(f"   ⚠️  GroupScore {groupscore:.4f} 低于阈值 {self.reflection_threshold}")
                
                # 提取分数低的链接
                low_score_links = []
                if hasattr(eval_result, 'pair_scores') and eval_result.pair_scores:
                    for pair_score in eval_result.pair_scores:
                        if pair_score.get('combined_score', 0) < self.reflection_threshold:
                            low_score_links.append({
                                "title": pair_score.get('link_title', ''),
                                "platform": pair_score.get('link_platform', ''),
                                "score": pair_score.get('combined_score', 0)
                            })
                
                if low_score_links:
                    print(f"   📉 发现 {len(low_score_links)} 个低相关性链接:")
                    for link in low_score_links[:3]:  # 只显示前3个
                        print(f"      - {link['title'][:50]}... [分数: {link['score']:.4f}]")
                
                # 4. 应用改进（移除低相关性链接 + 重新生成新链接）
                print(f"\n4️⃣  应用改进...")
                result = self._apply_reflection_suggestions_hupu(
                    current_html_path,
                    eval_result,
                    self.reflection_threshold,
                    iteration,
                    text_content=text,
                    hot_topics=hot_topics,
                    removed_links_history=removed_links_history,
                    user_profile=user_profile  # 添加user_profile参考
                )
                
                # 解析返回值（可能是元组或字典）
                if isinstance(result, tuple):
                    new_html_path, newly_removed = result
                    new_links_added = 0  # 默认值，如果函数返回了更多信息需要更新
                    text_optimized = False
                else:
                    # 如果返回字典，提取信息
                    new_html_path = result.get('html_path')
                    newly_removed = result.get('removed_links', [])
                    new_links_added = result.get('new_links_added', 0)
                    text_optimized = result.get('text_optimized', False)
                
                # 更新删除历史
                if newly_removed:
                    removed_links_history.extend(newly_removed)
                
                if new_html_path:
                    # 优化验证：只在有明显改进预期时才验证（减少评估次数）
                    # 策略：如果添加了新链接或优化了文本，才进行验证
                    # 如果只是删除了链接，直接采用（删除低分链接通常不会降低分数）
                    should_verify = True
                    verify_reason = "修改后验证"
                    
                    # 检查是否只是删除链接（没有添加新链接或优化文本）
                    if new_links_added == 0 and not text_optimized:
                        # 只删除链接，通常不会降低分数，可以跳过验证
                        should_verify = False
                        verify_reason = "仅删除链接，跳过验证（通常不会降低分数）"
                    
                    if should_verify:
                        print(f"\n5️⃣  验证新版本效果...")
                        try:
                            new_eval_result = evaluate_file_links(
                                html_path=new_html_path,
                                evaluator=self.text_evaluator,
                                verbose=False
                            )
                            new_groupscore = new_eval_result.group_score_mean
                            score_delta = new_groupscore - groupscore
                            
                            print(f"   📊 修改前: {groupscore:.4f}")
                            print(f"   📊 修改后: {new_groupscore:.4f}")
                            print(f"   📊 变化: {score_delta:+.4f}")
                            
                            if new_groupscore >= groupscore:
                                # 分数提升或持平，采用新版本
                                current_html_path = new_html_path
                                print(f"   ✅ 分数{'提升' if score_delta > 0 else '持平'}，采用新版本")
                            else:
                                # 分数下降，不采用新版本
                                print(f"   ⚠️  分数下降，保留原版本")
                                # 删除新生成的HTML文件
                                if os.path.exists(new_html_path):
                                    os.remove(new_html_path)
                        except Exception as e:
                            print(f"   ⚠️  验证失败: {e}")
                            # 验证失败，保守起见不采用新版本
                            if os.path.exists(new_html_path):
                                os.remove(new_html_path)
                    else:
                        # 跳过验证，直接采用新版本
                        print(f"\n5️⃣  {verify_reason}，直接采用新版本")
                        current_html_path = new_html_path
                else:
                    # 第3次及以后的反思，如果修改失败，继续下一次迭代
                    # 前3次如果修改失败，可能真的没有需要改进的地方，可以停止
                    if iteration >= 2:
                        iteration_num = iteration + 1
                        print(f"   ℹ️  无需修改或修改失败，继续下一次迭代（第{iteration_num}次反思）")
                        # 继续下一次迭代，不break
                        continue
                    else:
                        print(f"   ℹ️  无需修改或修改失败，停止Reflection。")
                        break
            
            # 最后一次迭代后，再评估一次最终版本的分数
            if current_html_path != html_path:  # 如果有生成新版本
                print(f"\n📊 评估最终版本...")
                try:
                    final_eval_result = evaluate_file_links(
                        html_path=current_html_path,
                        evaluator=self.text_evaluator,
                        verbose=False
                    )
                    final_groupscore = final_eval_result.group_score_mean  # 使用算术平均
                    print(f"   ✅ 最终 GroupScore (Mean): {final_groupscore:.4f}")
                    print(f"      (Harmonic: {final_eval_result.group_score_harmonic:.4f})")
                    
                    # 更新最后一个版本的分数（如果是break出来的，已经有了；如果是最后一次迭代，需要更新）
                    if reflection_history and reflection_history[-1]["html_path"] == current_html_path:
                        reflection_history[-1]["groupscore"] = final_groupscore
                    else:
                        # 添加最终版本记录
                        reflection_history.append({
                            "version": f"v{len(reflection_history)}",
                            "groupscore": final_groupscore,
                            "html_path": current_html_path
                        })
                except Exception as e:
                    print(f"   ⚠️  最终评估失败: {e}")
            
            # 选择分数最高的版本
            if reflection_history:
                best_version = max(reflection_history, key=lambda x: x.get("groupscore", 0))
                best_html_path = best_version["html_path"]
                best_score = best_version["groupscore"]
                best_version_name = best_version["version"]
                
                print(f"\n📊 Reflection总结:")
                print(f"   - 总迭代次数: {len(reflection_history)}")
                print(f"   - 最高分版本: {best_version_name} (分数: {best_score:.4f})")
                for hist in reflection_history:
                    indicator = "👑" if hist["html_path"] == best_html_path else "  "
                    print(f"   {indicator} {hist['version']}: {hist['groupscore']:.4f}")
                
                current_html_path = best_html_path
                print(f"\n✅ 选择版本 {best_version_name} 用于最终输出和英文转换")
            
            print("\n" + "="*80)
            print(f"✅ Reflection机制结束。最终版本: {os.path.basename(current_html_path)}")
            print("="*80)
            
            # 更新html_path为最终版本（最高分版本）
            html_path = current_html_path
        elif self.reflection_enabled and use_fallback_mode:
            print("\n⚠️  Fallback模式无链接，跳过Reflection评估")
        elif self.reflection_enabled and not links:
            print("\n⚠️  本帖无链接，跳过Reflection评估")
        # ========================= End of Reflection =========================
        
        # 构建返回结果
        result = {
            "text": text, 
            "images": images, 
            "links": links, 
            "hot_topics": hot_topics if not use_fallback_mode else [],
            "ideas": ideas,
            "html_post": html_path,
            "style": style,
            "mode": "fallback" if use_fallback_mode else "hot_topics",
            "reflection_history": reflection_history if reflection_history else None
        }
        
        # 如果是降级模式，添加说明
        if use_fallback_mode:
            result["note"] = "Generated from user profile (no hot topics available)"
        
        return result
    
    def _apply_reflection_suggestions_hupu(self, html_path, eval_result, threshold, iteration, 
                                           text_content=None, hot_topics=None, removed_links_history=None,
                                           user_profile=None):
        """
        应用Reflection改进建议（虎扑版本：移除低相关性链接 + 重新生成新链接）
        
        Args:
            html_path: 当前HTML文件路径
            eval_result: 评估结果（GroupScoreResult对象）
            threshold: 相关性阈值
            iteration: 当前迭代次数
            text_content: 帖子文本内容（用于重新筛选相关链接）
            hot_topics: 热点话题列表（用于重新生成链接）
            removed_links_history: 已删除链接的历史记录（避免重复添加）
            user_profile: 用户画像（用于重新搜索热点话题时参考）
            
        Returns:
            (新HTML文件路径, 本次删除的链接列表)，如果无需修改则返回(None, [])
        """
        if removed_links_history is None:
            removed_links_history = []
        from bs4 import BeautifulSoup
        
        try:
            # 读取当前HTML
            with open(html_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # 找到所有链接
            content_div = soup.find('div', class_='post-content')
            if not content_div:
                print("   ⚠️  未找到post-content div")
                return {
                    'html_path': None,
                    'removed_links': [],
                    'new_links_added': 0,
                    'text_optimized': False
                }
            
            link_cards = content_div.find_all('a', class_='link-card')
            
            if not link_cards:
                print("   ⚠️  未找到任何链接")
                return {
                    'html_path': None,
                    'removed_links': [],
                    'new_links_added': 0,
                    'text_optimized': False
                }
            
            # 分析哪些链接需要移除
            links_to_remove = []
            if hasattr(eval_result, 'pair_scores') and eval_result.pair_scores:
                for i, pair_score in enumerate(eval_result.pair_scores):
                    combined_score = pair_score.get('combined_score', 0)
                    if combined_score < threshold:
                        link_title = pair_score.get('link_title', '')
                        links_to_remove.append({
                            "index": i,
                            "title": link_title,
                            "score": combined_score
                        })
            
            if not links_to_remove:
                print("   ℹ️  所有链接相关性均达标，无需移除")
                return {
                    'html_path': None,
                    'removed_links': [],
                    'new_links_added': 0,
                    'text_optimized': False
                }
            
            print(f"   🗑️  准备移除 {len(links_to_remove)} 个低相关性链接:")
            for link_info in links_to_remove:
                print(f"      - {link_info['title'][:50]}... [分数: {link_info['score']:.4f}]")
            
            # 先记录要删除的链接信息（但先不删除，等确认有新链接可替代）
            removed_count = 0
            newly_removed_titles = []
            links_to_remove_elements = []
            
            for link_info in links_to_remove:
                idx = link_info['index']
                if idx < len(link_cards):
                    link_card = link_cards[idx]
                    # 记录链接信息
                    title_div = link_card.find('div', class_='link-title')
                    if title_div:
                        link_title = title_div.get_text(strip=True)
                        link_url = link_card.get('href', '')
                        newly_removed_titles.append({
                            "title": link_title,
                            "url": link_url,
                            "element": link_card  # 保存元素引用
                        })
                        links_to_remove_elements.append(link_card)
            
            print(f"   ⏸️  暂存 {len(newly_removed_titles)} 个待处理链接")
            
            # =================== 策略选择：添加新链接 OR 优化文本 ===================
            new_links_added = 0
            text_optimized = False
            
            if text_content and hot_topics:
                print(f"\n   🔄 尝试重新生成更相关的链接...")
                
                try:
                    # 1. 筛选相关话题
                    relevant_topics = self._filter_relevant_links(text_content, hot_topics)
                    print(f"      ✅ 筛选出 {len(relevant_topics)} 个相关话题")
                    
                    # 2. 生成新链接
                    new_links = self._extract_links_from_topics(relevant_topics)
                    print(f"      ✅ 生成 {len(new_links)} 个新链接候选")
                    
                    # 3. 获取已存在的链接和黑名单
                    remaining_links = content_div.find_all('a', class_='link-card')
                    existing_titles = set()
                    for link in remaining_links:
                        title_div = link.find('div', class_='link-title')
                        if title_div:
                            existing_titles.add(title_div.get_text(strip=True))
                    
                    # 黑名单（使用URL作为唯一标识）
                    removed_urls_set = set()
                    for removed in removed_links_history:
                        if removed.get('url'):
                            removed_urls_set.add(removed['url'])
                    for removed in newly_removed_titles:
                        if removed.get('url'):
                            removed_urls_set.add(removed['url'])
                    
                    print(f"      ℹ️  黑名单: {len(removed_urls_set)} 个历史删除的链接URL")
                    
                    # 4. 筛选可用的新链接
                    available_new_links = []
                    for new_link in new_links:
                        link_title = new_link.get('title', '')
                        link_url = new_link.get('url', '')
                        
                        is_blacklisted = link_url in removed_urls_set if link_url else False
                        
                        if is_blacklisted:
                            print(f"      🚫 过滤黑名单: {link_title[:50]}...")
                        elif link_title in existing_titles:
                            print(f"      ⏭️  跳过重复: {link_title[:50]}...")
                        else:
                            available_new_links.append(new_link)
                    
                    # 5. 根据是否有可用新链接和迭代次数选择策略
                    if available_new_links:
                        # 策略A: 有可用新链接 → 删除旧链接，添加新链接（所有迭代都可用）
                        iteration_num = iteration + 1
                        print(f"\n   ✅ 发现 {len(available_new_links)} 个可用新链接")
                        print(f"   🗑️  确认删除旧链接...")
                        
                        # 真正删除旧链接
                        for link_elem in links_to_remove_elements:
                            link_elem.decompose()
                            removed_count += 1
                        
                        print(f"   ➕ 添加 {len(available_new_links)} 个新链接...")
                        for new_link in available_new_links:
                            link_title = new_link.get('title', '')
                            link_html = self._create_link_tag(new_link)
                            link_soup = BeautifulSoup(link_html, 'html.parser')
                            content_div.append(link_soup)
                            new_links_added += 1
                            print(f"      ✅ {link_title[:50]}...")
                    elif iteration == 0:
                        # 策略B: 无可用新链接 + 第一次Reflection → 重新搜索新热点
                        # 只在第1次反思时允许重新搜索
                        iteration_num = iteration + 1
                        print(f"\n   ⚠️  没有可用的新链接（都在黑名单或已存在）")
                        print(f"   🔄 [第{iteration_num}次Reflection] 重新搜索新的热点话题...")
                        
                        try:
                            # 重新调用搜索API获取新的热点
                            import sys
                            # 使用真正的user_profile（如果提供），否则使用text_content推断
                            if user_profile:
                                search_profile = user_profile
                                print(f"      📋 使用用户画像重新搜索热点话题...")
                            else:
                                search_profile = f"用户兴趣：{text_content[:200]}..."
                                print(f"      ⚠️  未提供user_profile，使用文本内容推断...")
                            
                            new_hot_topics = self.search_hot_topics(search_profile)
                            
                            if new_hot_topics and new_hot_topics != "NO_VERIFIED_TRENDS_FOUND":
                                print(f"      ✅ 搜索到 {len(new_hot_topics)} 个新热点")
                                
                                # 从新热点中筛选相关的
                                new_relevant_topics = self._filter_relevant_links(text_content, new_hot_topics)
                                print(f"      ✅ 筛选出 {len(new_relevant_topics)} 个相关新话题")
                                
                                # 从新热点生成链接
                                additional_links = self._extract_links_from_topics(new_relevant_topics)
                                print(f"      ✅ 从新热点生成 {len(additional_links)} 个新链接")
                                
                                # 过滤黑名单和重复
                                for add_link in additional_links:
                                    link_title = add_link.get('title', '')
                                    link_url = add_link.get('url', '')
                                    
                                    is_blacklisted = link_url in removed_urls_set if link_url else False
                                    
                                    if is_blacklisted:
                                        print(f"         🚫 过滤黑名单: {link_title[:50]}...")
                                    elif link_title in existing_titles:
                                        print(f"         ⏭️  跳过重复: {link_title[:50]}...")
                                    else:
                                        # 真正删除旧链接（如果还没删除）
                                        if removed_count == 0:
                                            print(f"      🗑️  确认删除旧链接...")
                                            for link_elem in links_to_remove_elements:
                                                link_elem.decompose()
                                                removed_count += 1
                                        
                                        # 添加新链接
                                        link_html = self._create_link_tag(add_link)
                                        link_soup = BeautifulSoup(link_html, 'html.parser')
                                        content_div.append(link_soup)
                                        new_links_added += 1
                                        existing_titles.add(link_title)
                                        print(f"         ➕ 添加: {link_title[:50]}...")
                                
                                if new_links_added > 0:
                                    print(f"      ✅ 从新热点添加 {new_links_added} 个链接")
                                else:
                                    print(f"      ⚠️  新热点的链接也都不可用，转而优化文本...")
                                    # 转到策略C
                                    removed_count = 0
                                    newly_removed_titles = []
                            else:
                                print(f"      ⚠️  搜索新热点失败，转而优化文本...")
                                # 转到策略C（会在下面的else中处理）
                        
                        except Exception as e:
                            print(f"      ⚠️  搜索新热点异常: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # 策略C: 最后手段 → 保留原链接，优化文本
                    # 第2次及以后（iteration >= 1），如果没有可用新链接，直接使用策略C
                    # 第3次及以后（iteration >= 2），如果没有可用新链接，也使用策略C（重复第3次策略）
                    if new_links_added == 0 and removed_count == 0:
                        iteration_num = iteration + 1
                        if iteration >= 2:
                            print(f"\n   💡 [第{iteration_num}次Reflection] 使用第3次策略: 保留原链接，优化文本使其与链接更相关...")
                        else:
                            print(f"\n   💡 [第{iteration_num}次Reflection] 最后策略: 保留原链接，优化文本使其与链接更相关...")
                        
                        # 不删除链接，保持现状
                        removed_count = 0
                        newly_removed_titles = []  # 清空删除列表
                        
                        # 提取当前所有链接信息用于文本优化
                        current_links_for_optimization = []
                        for link_card in link_cards:
                            title_div = link_card.find('div', class_='link-title')
                            if title_div:
                                link_title = title_div.get_text(strip=True)
                                current_links_for_optimization.append({
                                    "title": link_title,
                                    "url": link_card.get('href', '')
                                })
                        
                        if current_links_for_optimization:
                            print(f"      📝 优化目标: 使文本与以下 {len(current_links_for_optimization)} 个链接更相关")
                            for link in current_links_for_optimization[:2]:
                                print(f"         - {link['title'][:50]}...")
                            
                            # 调用AI优化文本
                            optimized_paragraphs = self._optimize_text_for_links_hupu(
                                soup, 
                                content_div, 
                                current_links_for_optimization[:2],
                                hot_topics
                            )
                            
                            if optimized_paragraphs > 0:
                                text_optimized = True
                                print(f"   ✅ 已优化 {optimized_paragraphs} 个段落，保留 {len(current_links_for_optimization)} 个链接")
                            else:
                                print(f"   ⚠️  文本优化失败")
                    
                except Exception as e:
                    print(f"   ⚠️  处理失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   ⚠️  缺少text_content或hot_topics，无法处理")
            
            # 检查最终链接数量和优化状态
            final_links = content_div.find_all('a', class_='link-card')
            action_desc = f"移除{removed_count}个"
            if new_links_added > 0:
                action_desc += f"，添加{new_links_added}个"
            if text_optimized:
                action_desc += f"，优化文本"
            print(f"   📊 最终链接数: {len(final_links)} ({action_desc})")
            
            if not final_links:
                print(f"   ⚠️  当前无链接（可能在后续迭代中添加）")
            
            # 保存新版本
            output_dir = Path(html_path).parent
            version = f"_v{iteration+1}"  # v1, v2, v3
            new_html_path = output_dir / f"discussion_post{version}.html"
            
            with open(new_html_path, 'w', encoding='utf-8') as f:
                f.write(str(soup.prettify()))
            
            print(f"   ✅ 新版本已保存: {new_html_path.name}")
            # 返回详细信息，包括修改类型
            return {
                'html_path': str(new_html_path),
                'removed_links': newly_removed_titles,
                'new_links_added': new_links_added,
                'text_optimized': text_optimized
            }
            
        except Exception as e:
            print(f"   ❌ 应用改进失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'html_path': None,
                'removed_links': [],
                'new_links_added': 0,
                'text_optimized': False
            }
    
    def _optimize_text_for_links_hupu(self, soup, content_div, removed_links, hot_topics):
        """
        优化文本内容以提高与链接的相关性（Hupu版本）
        
        当无法生成新链接时，通过优化文本使其与已有话题更相关，从而提高GroupScore
        
        Args:
            soup: BeautifulSoup对象
            content_div: 内容容器div
            removed_links: 被删除的链接列表（包含title, url）
            hot_topics: 热点话题列表
            
        Returns:
            优化的段落数量
        """
        try:
            # 提取所有段落
            paragraphs = content_div.find_all('p')
            if not paragraphs:
                print(f"      ⚠️  未找到段落")
                return 0
            
            # 构建话题上下文
            topics_context = ""
            if removed_links:
                topics_context = "需要提高相关性的话题:\n"
                for link in removed_links:
                    topics_context += f"- {link['title']}\n"
            
            # 找到与话题相关的热点详情
            topic_details = ""
            if hot_topics and isinstance(hot_topics, list):
                topic_details = "\n相关热点详情:\n"
                for topic in hot_topics[:3]:
                    if isinstance(topic, dict):
                        topic_details += f"- {topic.get('topic', '')}\n"
            
            # 优化前两个段落（通常是引言和主要观点）
            optimized_count = 0
            for i, para in enumerate(paragraphs[:2]):
                old_text = para.get_text(strip=True)
                if not old_text or len(old_text) < 20:
                    continue
                
                print(f"      🔄 优化段落 {i+1}...")
                
                # 构建优化prompt
                optimize_prompt = f"""你是虎扑论坛的资深用户。请优化以下讨论帖的段落，使其与相关话题更紧密结合。

{topics_context}

{topic_details}

**原始段落：**
{old_text}

**优化要求：**
1. 保持虎扑论坛风格（直接、有观点、接地气）
2. 在段落中自然融入与上述话题相关的讨论点
3. 可以提及具体的球员、球队、数据等细节
4. 保持段落长度相近（不要过度扩写）
5. 增强与话题的关联性，但要自然，不要生硬

**只输出优化后的段落文字，不要任何解释。**"""
                
                try:
                    resp = requests.post(
                        f"{self.search_base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.search_api_key}"},
                        json={
                            "model": self.search_model,
                            "messages": [{"role": "user", "content": optimize_prompt}],
                            "temperature": 0.7,
                            "max_tokens": 500
                        },
                        timeout=30
                    )
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        new_text = result['choices'][0]['message']['content'].strip()
                        
                        if new_text and new_text != old_text:
                            # 替换段落内容
                            para.clear()
                            para.string = new_text
                            optimized_count += 1
                            print(f"         ✅ 已优化 (长度: {len(old_text)} → {len(new_text)})")
                        else:
                            print(f"         ⏭️  无变化")
                    else:
                        print(f"         ⚠️  API错误: {resp.status_code}")
                        
                except Exception as e:
                    print(f"         ⚠️  优化失败: {e}")
                    continue
            
            return optimized_count
            
        except Exception as e:
            print(f"      ❌ 文本优化异常: {e}")
            import traceback
            traceback.print_exc()
            return 0

