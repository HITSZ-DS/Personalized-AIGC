import json
import os
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from tenacity import retry


SYSTEM_PROMPT_PROFILE_TO_IDEA = \
"""
Note: You are a creative AI specialized in generating innovative product ideas for Xiaohongshu (小红书) text-image posts based on user profiles. Your task is to create unique and feasible product ideas that align with the user's interests and preferences, following these requirements:

1. Your product types should be selected from the following product types (ALL are Text-Image Content for Xiaohongshu platform):  
{product_types}  
2. Each product idea should be concise and clear, yet possess a distinct theme.  
3. Each idea should have relevant supporting evidence from the user profile.
4. The number of product ideas should be 2-3.
5. IMPORTANT: ALL ideas MUST be "Text-Image Content" type (main_type must be "Text-Image Content"). DO NOT generate any video content ideas.

**Output Format**:
Please return the response to me in the following format (pure JSON only, do NOT wrap with markdown code blocks):
[
  {{
    "idea": "Product Idea 1",
    "main_type": "Text-Image Content",
    "type": "Product Type",
    "basis": "Supporting evidence from user profile"
  }},
  {{
    "idea": "Product Idea 2",
    "main_type": "Text-Image Content", 
    "type": "Product Type",
    "basis": "Supporting evidence from user profile"
  }},
  ...
]

IMPORTANT: Return ONLY the JSON array without any markdown formatting (no ```json or ``` tags).
"""

PRODUCT_TYPES = \
"""
Main Type: Text-Image Content (小红书图文帖子)

Type 1: News Updates (资讯速递)
Description: Brief text (usually under 500 words) with intuitive images, focusing on quick dissemination and instant interaction. Commonly used for hot topics, news, or personal updates.

Type 2: Deep Analysis (深度分析)
Description: Focus on deep content and rigorous logic, emphasizing opinion sharing, industry insights, or professional knowledge. Aims to deliver professional value.

Type 3: Lifestyle Guides (生活方式指南)
Description: High-quality images (often in 9-grid format) with text highlighting practicality and real experiences. Core concept is "documenting my life" to provide reference for others' consumption decisions.

Type 4: Product Reviews (产品测评)
Description: Professional, clear, and in-depth content providing decision-making basis through personal testing and detailed comparisons. Common in tech, appliances, etc.

Type 5: Community Discussion (社区讨论)
Description: Deep discussions and creations around specific interests (like books, films, music, hobby groups). Strong community atmosphere with emphasis on authentic expression.

Type 6: Tutorial Guides (教程指南)
Description: Focus on clear step demonstration to help users "how to accomplish something". Requires logical structure and strong practicality.

Type 7: Collection Lists (合集清单)
Description: Compiling similar items or information systematically, such as product collections, book/movie lists, helping users quickly access systematic information.

Type 8: Opinion Posts (观点分享)
Description: Sharing personal insights and deep thoughts on social events or topics to spark discussion and resonance. Emphasizes logic and intellectual depth.

Type 9: Interactive Posts (互动帖)
Description: Actively inviting user participation through polls, topics, Q&A, etc. Aims to increase user engagement and interaction metrics.

Type 10: Story Content (故事内容)
Description: Publishing longer personal stories or fiction online to attract readers and create empathy through storytelling.
"""

class Profile2Idea:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("CHAT_API_KEY"),
            base_url=os.getenv("CHAT_BASE_URL"),
        )

    def __call__(self, user_profile: str, user_dir: str):
        print("user profile: ", user_profile)
        idea_path = os.path.join(user_dir, "product_ideas.json")
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=SYSTEM_PROMPT_PROFILE_TO_IDEA.format(product_types=PRODUCT_TYPES)
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=user_profile
            )
        ]

        print("Generating Product Ideas...")
        completion = self.client.chat.completions.create(
            model=os.getenv("CHAT_MODEL"),
            messages=messages
        )
        
        raw_content = completion.choices[0].message.content
        
        # 清洗输出：移除可能的 markdown 代码块标记
        cleaned_content = raw_content.strip()
        
        # 移除 ```json 和 ``` 标记
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        elif cleaned_content.startswith("```"):
            cleaned_content = cleaned_content[3:]
        
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]
        
        cleaned_content = cleaned_content.strip()
        
        # 保存清洗后的内容
        with open(idea_path, "w", encoding="utf-8") as f:
            try:
                # 尝试解析验证 JSON 格式
                parsed_json = json.loads(cleaned_content)
                # 如果成功，保存格式化的 JSON
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                # 如果失败，保存原始内容（用于调试）
                f.write(cleaned_content)
        
        print(cleaned_content)
        print("Product Ideas generated.")
        return cleaned_content



