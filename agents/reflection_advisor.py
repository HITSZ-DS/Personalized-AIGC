"""
Reflection Advisor for ITProduct
为小红书图文笔记提供 AI 驱动的反思和改进建议

输入：
1. GroupScore（图文一致性分数）
2. HTML解析序列（文本+图片+链接的排布）
3. RAG Top-3 优秀样例

输出：
- 是否需要修改的判断
- 文本修改建议
- 图片Caption建议
"""

import os
import json
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional


class ReflectionAdvisor:
    """AI反思顾问 - 分析内容并提供改进建议"""
    
    def __init__(self):
        # 使用搜索模型进行反思（支持联网和深度分析）
        self.api_key = os.getenv("SEARCH_API_KEY", "sk-dVaSXmTEMBh0Gygx49ResSvaONvErml5QV8McBAGkbPmX2mG")
        self.api_base = os.getenv("SEARCH_BASE_URL", "https://yunwu.ai/v1")
        self.model = os.getenv("SEARCH_MODEL", "gpt-5-all")
        
        # 多模态模型配置（用于图片caption生成）
        self.vision_model = os.getenv("VISION_MODEL", "claude-sonnet-4-5-20250929")  # Claude 3.5 Sonnet多模态
        
        if not self.api_key:
            raise ValueError("SEARCH_API_KEY not found in environment variables")
    
    def evaluate_and_suggest(
        self,
        groupscore: float,
        html_sequence: str,
        rag_examples: List[Dict],
        image_paths: List[str] = None,
        threshold: float = 0.65,
        user_profile: Optional[str] = None,
        reflection_iteration: int = 0
    ) -> Dict:
        """
        综合评估内容并生成改进建议
        
        Args:
            groupscore: CLIP 评分（图文一致性）
            html_sequence: 解析后的HTML序列文本
            rag_examples: RAG检索的优秀样例（Top-3）
            image_paths: 图片文件路径列表（用于多模态分析）
            threshold: GroupScore 阈值（默认0.65）
            user_profile: 用户画像（可选）
            
        Returns:
            {
                "should_modify": bool,
                "reason": str,
                "suggestions": {
                    "text_changes": [...],
                    "image_captions": [...]
                },
                "image_analyses": [...],  # 多模态模型生成的图片caption
                "overall_assessment": str
            }
        """
        
        print(f"\n🤖 AI Reflection Advisor 启动...")
        print(f"   GroupScore: {groupscore:.4f} (阈值: {threshold})")
        
        # 1. 初步判断：GroupScore 是否低于阈值
        if groupscore >= threshold:
            print(f"   ✅ GroupScore 达标，无需 Reflection")
            return {
                "should_modify": False,
                "reason": f"GroupScore ({groupscore:.4f}) 已达到阈值 ({threshold})",
                "suggestions": None,
                "overall_assessment": "内容质量良好"
            }
        
        print(f"   ⚠️  GroupScore 低于阈值，启动深度分析...")
        
        # 2. 分析图片（如果有图片且提供了路径）
        image_analyses = []
        if image_paths and len(image_paths) > 0:
            try:
                # 提取部分文本作为context
                context_text = html_sequence[:500] if html_sequence else ""
                image_analyses = self._analyze_images_with_vision(image_paths, context_text)
            except Exception as e:
                print(f"   ⚠️  图片分析失败: {e}")
                image_analyses = []
        else:
            print(f"   ℹ️  无图片需要分析")
        
        # 3. 格式化 RAG 样例
        examples_text = self._format_rag_examples(rag_examples)
        
        # 4. 格式化图片caption（如果有）
        image_caption_text = ""
        if image_analyses:
            image_caption_text = "\n**多模态模型生成的图片Caption：**\n"
            for caption_item in image_analyses:
                image_caption_text += f"- {caption_item['caption']}\n"
        
        # 5. 构建评估 Prompt
        evaluation_prompt = self._build_evaluation_prompt(
            groupscore=groupscore,
            html_sequence=html_sequence,
            examples_text=examples_text,
            threshold=threshold,
            user_profile=user_profile,
            image_caption=image_caption_text,
            reflection_iteration=reflection_iteration
        )
        
        # 6. 调用 AI 进行评估和建议
        try:
            reflection_result = self._call_ai_for_reflection(evaluation_prompt)
            
            # 7. 解析结果
            parsed_result = self._parse_reflection_result(reflection_result)
            
            # 8. 将图片分析结果整合到返回中
            parsed_result["image_analyses"] = image_analyses
            
            print(f"\n📋 AI 评估结果:")
            print(f"   是否需要修改: {'是' if parsed_result['should_modify'] else '否'}")
            print(f"   理由: {parsed_result['reason'][:100]}...")
            
            if parsed_result['should_modify'] and parsed_result['suggestions']:
                text_changes = parsed_result['suggestions'].get('text_changes', [])
                captions = parsed_result['suggestions'].get('image_captions', [])
                print(f"   文本修改建议: {len(text_changes)} 条")
                print(f"   图片Caption建议: {len(captions)} 条")
            
            return parsed_result
            
        except Exception as e:
            print(f"   ❌ AI 评估失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 降级策略：如果AI调用失败，使用多模态生成的caption（如果有）
            fallback_captions = []
            if image_analyses:
                fallback_captions = [
                    {
                        "position": f"image_{caption_item['image_index']}",
                        "caption": caption_item['caption']
                    }
                    for caption_item in image_analyses
                ]
            
            return {
                "should_modify": True,
                "reason": f"GroupScore ({groupscore:.4f}) 低于阈值，AI评估失败但建议修改",
                "suggestions": {
                    "text_changes": [
                        {
                            "position": "整体",
                            "issue": "图文一致性不足",
                            "suggestion": "建议优化文案与图片的匹配度"
                        }
                    ],
                    "image_captions": fallback_captions
                },
                "image_analyses": image_analyses,
                "overall_assessment": "需要改进（AI评估失败，使用降级方案 + 多模态图片分析）"
            }
    
    def _format_rag_examples(self, rag_examples: List[Dict]) -> str:
        """格式化RAG样例为文本"""
        if not rag_examples:
            return "（无优秀样例参考）"
        
        formatted = []
        for i, example in enumerate(rag_examples[:3], 1):  # 最多3个
            content = example.get('content', example.get('full_text', ''))
            similarity = example.get('similarity', 0)
            
            # 截断过长的内容
            if len(content) > 500:
                content = content[:500] + "..."
            
            formatted.append(f"【优秀样例 {i}】(相似度: {similarity:.3f})\n{content}")
        
        return "\n\n".join(formatted)
    
    def _build_evaluation_prompt(
        self,
        groupscore: float,
        html_sequence: str,
        examples_text: str,
        threshold: float,
        user_profile: Optional[str],
        image_caption: str = "",
        reflection_iteration: int = 0
    ) -> str:
        """构建评估Prompt"""
        
        # 用户画像部分（如果有）
        profile_section = ""
        if user_profile:
            profile_section = f"""
**用户画像：**
{user_profile[:300]}
"""
        
        prompt = f"""You are a Xiaohongshu content quality expert specializing in image-text semantic consistency optimization. Your primary goal is to MAXIMIZE the GroupScore (CLIP-based image-text matching) through strategic improvements.

## ⚠️ CRITICAL: Caption-Only Evaluation

**GroupScore is calculated ONLY using image captions and images** (not full text paragraphs).
- CLIP model is trained on simple image-text pairs, not long paragraphs
- Only caption text is used for similarity calculation
- Full paragraph text is NOT used in GroupScore calculation

## Evaluation Materials

**1. Image-Text Consistency Score (GroupScore): {groupscore:.4f} / 1.0**
   - Threshold: {threshold}
   - Status: {'⚠️ Below threshold' if groupscore < threshold else '✅ Meets threshold'}
   - Explanation: CLIP model evaluates semantic similarity between images and captions ONLY
   - **Goal**: Achieve 0.1+ improvement (target score: {groupscore + 0.1:.4f})

**2. Current Content Sequence:**
```
{html_sequence}
```

**3. High-Quality Examples (RAG Top-3):**
```
{examples_text}
```

{image_caption}

{profile_section}

## Evaluation Task

Analyze this post and provide strategic improvements to BOOST the GroupScore by 0.1+.

**You must base your evaluation on FOUR key sources:**
1. **GroupScore** ({groupscore:.4f}): Current image-caption consistency score - identify what's missing
2. **HTML Sequence**: Current content structure and caption-image relationships
3. **RAG Top-3 Examples**: High-quality reference posts to learn from
4. **User Profile**: User preferences and interests to ensure content alignment

### Evaluation Focus (Priority Order):

1. **Caption-Image Alignment** (MOST CRITICAL for score improvement)
   - Do captions accurately describe what's visible in the images?
   - Are key objects, colors, scenes, or actions from images mentioned in captions?
   - Missing keywords: What visual elements are shown but not mentioned in captions?
   - Caption quality: Are captions concise (2-6 characters) with concrete nouns?

2. **Caption Precision**
   - Are captions too vague (e.g., "美食") or too specific (e.g., "抹茶茉莉拿铁配白色陶瓷杯")?
   - Should captions be more focused on core visual elements?

3. **Content Quality** (Secondary)
   - Is the text natural and engaging?
   - Any redundant or off-topic sections?

### Improvement Strategies (for maximum score boost):

{f"**🎯 REFLECTION ITERATION {reflection_iteration + 1} STRATEGY:**" if reflection_iteration < 3 else ""}
{f'''
**Iteration 1 (First Reflection) - Image Regeneration Focus:**
- **Primary Goal**: Regenerate images to better match context + captions
- **Key References**: Context text, current captions (PRIMARY), RAG-top3 examples, user_profile (SECONDARY)
- **Image Modification Suggestions**: Provide specific guidance on what visual elements should be changed/added/removed
- **Focus**: Improve image-caption alignment by regenerating images that better match the captions and context
- **Do NOT modify captions** - only suggest image changes

1. **Image Modification Suggestions** (PRIMARY for Iteration 1):
   - Analyze current images vs. captions and context
   - Identify visual elements that are missing or misaligned
   - Suggest specific changes: objects to add/remove, colors, composition, style, etc.
   - Reference: Use context text and captions as PRIMARY guide, RAG examples and user_profile as SECONDARY reference
   - Format: Provide detailed image modification guidance for each image

2. **Image Captions** (NOT modified in Iteration 1):
   - Keep existing captions unchanged
   - Only analyze if captions are appropriate for the suggested image changes
''' if reflection_iteration == 0 else ""}
{f'''
**Iteration 2 (Second Reflection) - Caption Adjustment Based on Images:**
- **Primary Goal**: Adjust captions based on the actual images to ensure captions describe the MAIN SUBJECT in images
- **Key References**: Current images (from Iteration 1) - analyze what's ACTUALLY visible
- **Focus**: Ensure captions accurately describe the PRIMARY visual element/subject in each image
- **Do NOT regenerate images** - only modify captions based on image analysis

1. **Image Captions** (PRIMARY for Iteration 2):
   - Analyze each image to identify the MAIN SUBJECT (what occupies the center/foreground)
   - Update captions to match the PRIMARY visual element in each image
   - Caption format: **图1: [2-6 characters with core visual concept]**
   - Must describe the MAIN SUBJECT visible in the image, not secondary elements
   - Keep it concise: 2-6 characters, focus on the main visual theme
   - Example: If image shows "latte coffee" prominently → caption should be "拿铁咖啡" / "latte coffee"
''' if reflection_iteration == 1 else ""}
{f'''
**Iteration 3 (Third Reflection) - Caption-First Reconstruction:**
- **Primary Goal**: First generate concise captions, then regenerate images based on captions + partial context
- **Workflow**: 
  1. Generate new concise captions based on current images
  2. Regenerate images with captions as PRIMARY subject, partial context as SECONDARY reference
- **Key References**: 
  - Step 1: Current images (to extract main subjects for new captions)
  - Step 2: New captions (PRIMARY) + Partial context text (SECONDARY) + RAG-top3 + user_profile

1. **Image Captions** (Step 1 - Generate first):
   - Analyze current images to extract main subjects
   - Generate concise captions (2-6 characters) that describe the main visual element
   - Caption format: **图1: [2-6 characters with core visual concept]**

2. **Image Regeneration** (Step 2 - Based on new captions):
   - Regenerate images with NEW captions as the PRIMARY and DOMINANT subject
   - Use partial context text only for style, atmosphere, and background (SECONDARY)
   - Composition: Caption subject (70%+) > Context elements (30%-)
   - This ensures maximum CLIP score matching
''' if reflection_iteration == 2 else ""}
{f'''
1. **Image Captions** (ONLY way to improve GroupScore):
   - Multimodal model has generated captions (see above)
   - You can refine these captions to be more semantically precise
   - Caption format: **图1: [2-6 characters with core visual concept]**
   - Must include: Core concept, key objects (e.g., "抹茶饮品", "咖啡店", "好物分享")
   - Keep it concise: 2-6 characters, focus on the main visual theme
''' if reflection_iteration >= 3 else ""}

## Return Format (MUST be valid JSON)

{f'''
```json
{{
  "should_modify": true/false,
  "reason": "Brief reason (1-2 sentences, focus on image-caption alignment impact)",
  "suggestions": {{
    "text_changes": [],
    "image_modifications": [
      {{
        "position": "image_0",
        "current_issue": "What's wrong with current image",
        "suggested_changes": "Detailed guidance on visual elements to change/add/remove (objects, colors, composition, style, etc.)",
        "reference_priority": "Context + Caption (PRIMARY), RAG-top3 + user_profile (SECONDARY)"
      }}
    ],
    "image_captions": []  // Keep empty for Iteration 1
  }},
  "overall_assessment": "Overall evaluation (1-2 sentences, estimate expected score improvement)"
}}
```
''' if reflection_iteration == 0 else ""}
{f'''
```json
{{
  "should_modify": true/false,
  "reason": "Brief reason (1-2 sentences, focus on caption-image alignment impact)",
  "suggestions": {{
    "text_changes": [],
    "image_captions": [
      {{
        "position": "image_0",
        "caption": "图1: [2-6 characters, core visual concept, e.g., '抹茶饮品', '咖啡店', '好物分享']"
      }}
    ]
  }},
  "overall_assessment": "Overall evaluation (1-2 sentences, estimate expected score improvement)"
}}
```
''' if reflection_iteration == 1 else ""}
{f'''
```json
{{
  "should_modify": true/false,
  "reason": "Brief reason (1-2 sentences, focus on full reconstruction impact)",
  "suggestions": {{
    "text_changes": [],
    "image_modifications": [
      {{
        "position": "image_0",
        "current_issue": "What's wrong with current image",
        "suggested_changes": "Detailed guidance on visual elements to change/add/remove"
      }}
    ],
    "image_captions": [
      {{
        "position": "image_0",
        "caption": "图1: [2-6 characters, core visual concept, e.g., '抹茶饮品', '咖啡店', '好物分享']"
      }}
    ]
  }},
  "overall_assessment": "Overall evaluation (1-2 sentences, estimate expected score improvement)"
}}
```
''' if reflection_iteration == 2 else ""}
{f'''
```json
{{
  "should_modify": true/false,
  "reason": "Brief reason (1-2 sentences, focus on caption-image alignment impact)",
  "suggestions": {{
    "text_changes": [],
    "image_captions": [
      {{
        "position": "image_0",
        "caption": "图1: [2-6 characters, core visual concept, e.g., '抹茶饮品', '咖啡店', '好物分享']"
      }}
    ]
  }},
  "overall_assessment": "Overall evaluation (1-2 sentences, estimate expected score improvement)"
}}
```
''' if reflection_iteration >= 3 else ""}

**Critical Guidelines**:
- ⚠️ **Caption-Only**: GroupScore is calculated ONLY using captions, NOT full text paragraphs
- If GroupScore ≥ {threshold}: Return should_modify: false ONLY if captions are already optimal
- If GroupScore < {threshold}: MUST provide caption improvements targeting 0.1+ score boost
- Captions: Must be concise (2-6 characters), focus on core visual concept
- Examples of good captions: "抹茶饮品", "咖啡店", "好物分享", "旅行vlog"
- Examples of bad captions: "美食" (too vague), "抹茶茉莉拿铁配白色陶瓷杯" (too long)
- Expected impact: Prioritize caption changes with highest expected score improvement
- Output ONLY JSON, no other text

Begin evaluation:

"""
        
        return prompt
    
    def _call_ai_for_reflection(self, prompt: str) -> str:
        """调用AI进行反思"""
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,  # 较低温度，更精确的分析
                    "top_p": 0.9
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text[:200]}")
            
            result = response.json()
            
            if "choices" not in result:
                raise Exception(f"Unexpected response format: {result}")
            
            content = result["choices"][0]["message"]["content"].strip()
            
            return content
            
        except Exception as e:
            print(f"⚠️ AI调用异常: {e}")
            raise
    
    def _analyze_images_with_vision(self, image_paths: List[str], context_text: str = "") -> List[Dict]:
        """
        使用多模态模型为图片生成caption
        
        Args:
            image_paths: 图片绝对路径列表
            context_text: 文章上下文（帮助理解图片在文章中的作用）
            
        Returns:
            [
                {
                    "image_index": 0,
                    "caption": "图1: 简短描述（10-20字）"
                },
                ...
            ]
        """
        print(f"\n🔍 使用 {self.vision_model} 为 {len(image_paths)} 张图片生成caption...")
        
        image_captions = []
        
        for idx, img_path in enumerate(image_paths):
            if not img_path or not Path(img_path).exists():
                print(f"   ⚠️  跳过图片 {idx}: 文件不存在")
                image_captions.append({
                    "image_index": idx,
                    "caption": f"图{idx+1}: [图片缺失]"
                })
                continue
            
            try:
                # 读取图片并编码为base64
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                
                # 判断图片格式
                ext = Path(img_path).suffix.lower()
                mime_type = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.webp': 'image/webp',
                    '.gif': 'image/gif'
                }.get(ext, 'image/jpeg')
                
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # 构建vision prompt（第一人称视角，像作者给图片加的说明）
                vision_prompt = f"""你是小红书博主，正在为自己的照片添加说明文字。用第一人称视角描述这张图片，像是在给朋友介绍。

**背景**：这是你小红书笔记中的第{idx+1}张图片。

**你的笔记内容片段**：
{context_text[:300]}...

**Caption要求**：
1. **第一人称视角**：用"我"、"这里"、"今天"等，像是你在分享自己的照片
2. **具体的视觉细节**：提到具体的物品、颜色、场景（不要只说"很美"、"很棒"）
3. **贴近文章主题**：呼应文章内容，体现你在做什么、看到什么
4. **自然口语化**：像对朋友说话一样，可以用"超级"、"真的"等语气词
5. **10-40字**：简短但有信息量

**Caption格式**：图{idx+1}: [你的第一人称描述]

**好的示例（第一人称、具体、有画面感）**：
- 图1: 我站在茶卡盐湖的天空之镜前，白色盐晶地面倒映着蓝天白云，超美的
- 图2: 这就是敦煌莫高窟的九层楼！红色木质建筑里的壁画色彩真的好鲜艳
- 图3: 今天在山姆买了一大车，左边是清洁用品，右边冷柜装满了冷冻食品和肉
- 图4: 这杯拿铁的心形拉花我爱了！木质桌子配白色陶瓷杯超有质感
- 图5: 穿上这件新买的毛衣试拍一张，米色的很温柔显白

**不好的示例（太客观、第三人称、空泛）**：
- 图1: 茶卡盐湖天空之镜倒映云层（❌ 太客观，像导游解说）
- 图2: 美丽的风景（❌ 太空泛，没有具体信息）
- 图3: 一个人在购物（❌ 没有细节，不够生动）

**只输出caption文字**，不要其他内容。
"""
                
                # 调用多模态模型（简化版，只生成caption）
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.vision_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": vision_prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{img_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "temperature": 0.3,  # Lower temperature for more precise, consistent captions
                        "max_tokens": 250  # Caption更详细，需要更多tokens
                    },
                    timeout=60
                )
                
                if response.status_code != 200:
                    raise Exception(f"API error {response.status_code}: {response.text[:200]}")
                
                result = response.json()
                caption = result["choices"][0]["message"]["content"].strip()
                
                # 清理可能的markdown或多余格式
                caption = caption.replace("```", "").replace("**", "").strip()
                
                # 如果没有"图X:"前缀，自动添加
                if not caption.startswith(f"图{idx+1}"):
                    caption = f"图{idx+1}: {caption}"
                
                image_captions.append({
                    "image_index": idx,
                    "caption": caption
                })
                
                print(f"   ✅ 图片 {idx}: {caption}")
                
            except Exception as e:
                print(f"   ❌ 图片 {idx} caption生成失败: {e}")
                image_captions.append({
                    "image_index": idx,
                    "caption": f"图{idx+1}: [生成失败]"
                })
        
        return image_captions
    
    def _parse_reflection_result(self, ai_response: str) -> Dict:
        """解析AI返回的JSON结果"""
        
        # 清理可能的markdown标记
        import re
        content = ai_response
        
        # 移除 ```json 和 ```
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # 尝试解析JSON
        try:
            result = json.loads(content)
            
            # 验证必需字段
            required_fields = ["should_modify", "reason", "overall_assessment"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # 如果需要修改，验证 suggestions
            if result["should_modify"]:
                if "suggestions" not in result:
                    result["suggestions"] = {
                        "text_changes": [],
                        "image_captions": []
                    }
                
                # 确保 suggestions 有正确的结构
                if "text_changes" not in result["suggestions"]:
                    result["suggestions"]["text_changes"] = []
                if "image_captions" not in result["suggestions"]:
                    result["suggestions"]["image_captions"] = []
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            print(f"原始内容: {content[:500]}...")
            
            # 尝试修复常见的JSON问题
            try:
                # 移除可能的BOM和控制字符
                content = content.encode('utf-8').decode('utf-8-sig').strip()
                content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
                
                # 修复 trailing commas
                content = re.sub(r',\s*}', '}', content)
                content = re.sub(r',\s*]', ']', content)
                
                result = json.loads(content)
                print("✅ JSON修复成功")
                return result
                
            except Exception as e2:
                print(f"❌ JSON修复失败: {e2}")
                raise ValueError(f"无法解析AI返回的JSON: {e}")


def test_reflection_advisor():
    """测试 ReflectionAdvisor"""
    import sys
    from html_parser_for_reflection import HTMLParserForReflection
    
    print("="*80)
    print("🧪 测试 Reflection Advisor")
    print("="*80)
    
    # 创建实例
    try:
        advisor = ReflectionAdvisor()
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        print("💡 请确保设置了 SEARCH_API_KEY 环境变量")
        return
    
    # 测试文件
    html_path = "generated_it/29_5726fe0950c4b401f76283be/image_text.html"
    
    if not os.path.exists(html_path):
        print(f"❌ 测试文件不存在: {html_path}")
        return
    
    print(f"\n📝 测试文件: {html_path}")
    
    # 1. 解析HTML
    print("\n1️⃣ 解析HTML...")
    parser = HTMLParserForReflection()
    parse_result = parser.parse_html_to_sequence(html_path)
    html_sequence = parse_result["sequence_text"]
    image_paths = parse_result["image_paths"]
    
    print(f"   ✅ 解析完成: {parse_result['stats']['texts']} 文本, {parse_result['stats']['images']} 图片, {parse_result['stats']['links']} 链接")
    print(f"   📸 图片文件: {image_paths}")
    
    # 2. 模拟 GroupScore
    groupscore = 0.5923  # 低于阈值的分数
    print(f"\n2️⃣ GroupScore: {groupscore:.4f}")
    
    # 3. 模拟 RAG 样例
    rag_examples = [
        {
            "content": "姐妹们！今天要分享一个超棒的咖啡店☕️\n\n📍店名：Manner Coffee\n地址：朝阳大悦城2楼\n\n环境超级好，适合拍照📷 招牌拿铁真的绝了！",
            "similarity": 0.85
        },
        {
            "content": "周末去了趟西湖，风景美到窒息🌅\n\n必打卡景点：\n1. 断桥残雪\n2. 雷峰塔\n3. 苏堤春晓\n\n记得穿汉服拍照，超出片！",
            "similarity": 0.78
        }
    ]
    
    print(f"\n3️⃣ RAG样例: {len(rag_examples)} 个")
    
    # 4. 调用 Reflection
    print(f"\n4️⃣ 调用 AI Reflection...")
    print("-"*80)
    
    result = advisor.evaluate_and_suggest(
        groupscore=groupscore,
        html_sequence=html_sequence,
        rag_examples=rag_examples,
        image_paths=image_paths,
        threshold=0.65,
        user_profile="热爱旅行和美食的年轻女性用户"
    )
    
    # 5. 显示结果
    print("\n" + "="*80)
    print("📊 Reflection 结果")
    print("="*80)
    
    print(f"\n是否需要修改: {'✅ 是' if result['should_modify'] else '❌ 否'}")
    print(f"理由: {result['reason']}")
    print(f"整体评价: {result['overall_assessment']}")
    
    if result['should_modify'] and result['suggestions']:
        suggestions = result['suggestions']
        
        if suggestions.get('text_changes'):
            print(f"\n📝 文本修改建议 ({len(suggestions['text_changes'])} 条):")
            for i, change in enumerate(suggestions['text_changes'], 1):
                print(f"   {i}. 位置: {change.get('position', 'N/A')}")
                print(f"      问题: {change.get('issue', 'N/A')}")
                print(f"      建议: {change.get('suggestion', 'N/A')}")
        
        if suggestions.get('image_captions'):
            print(f"\n🖼️  图片Caption建议 ({len(suggestions['image_captions'])} 条):")
            for i, caption in enumerate(suggestions['image_captions'], 1):
                print(f"   {i}. {caption.get('position', 'N/A')}: {caption.get('caption', 'N/A')}")
    
    # 显示多模态模型生成的caption
    if result.get('image_analyses'):
        print(f"\n🔍 多模态模型生成的Caption ({len(result['image_analyses'])} 张):")
        for caption_item in result['image_analyses']:
            print(f"   📸 {caption_item['caption']}")
    
    print("\n" + "="*80)
    print("✅ 测试完成")
    print("="*80)
    
    # 保存结果
    output_path = "reflection_test_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存到: {output_path}")


# if __name__ == "__main__":
#     test_reflection_advisor()

