import json
import os
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from tenacity import retry


SYSTEM_PROMPT_PROFILE_TO_IDEA = \
"""
Note: You are a creative AI specialized in generating innovative product ideas based on user profiles. Your task is to create unique and feasible product ideas that align with the user's interests and preferences, following these requirements:

1. Your product types should be selected from the following product types:  
{product_types}  
2. Each product idea should be concise and clear, yet possess a distinct theme.  
3. Each idea should have relevant supporting evidence from the user profile.
4. The number of product ideas should not exceed 3.

**Output Format**:
Please return the response to me in the following format:
[
  {{
    "idea": "Product Idea 1",
    "main_type": "Main Category Type",
    "type": "Product Type",
    "basis": "Supporting evidence from user profile"
  }},
  {{
    "idea": "Product Idea 2",
    "main_type": "Main Category Type", 
    "type": "Product Type",
    "basis": "Supporting evidence from user profile"
  }},
  ...
]
"""

PRODUCT_TYPES = \
"""
Main Type: Video Content

Type 1: Cross Talk
Description: Adapt audio content of talk shows into Chinese crosstalk

Type 2: Meme Video
Description: Create engaging and viral-worthy meme content by intelligently transforming video materials with AI-generated audio and visual effects.

Type 3: Video Overview
Description: Transform lengthy video event into concise, engaging video overview with accurate information extraction.

Type 4: Music Video
Description: Create comprehensive music videos by generating lyrics, synthesizing vocals, and matching visuals to create engaging musical content.

Type 5: Talk Show
Description: Adapt audio content of Chinese crosstalk into stand-up comedy talk shows.

Type 6: Beat-synced Editing
Description: Create dynamic, beat-synced video edits by analyzing music tracks and intelligently cutting and enhancing video footage to match the rhythm and mood of the music.

Type 7: Storytelling Video
Description: Transform text-based stories into engaging storytelling videos by generating relevant visuals, narration, and sound effects to bring the story to life.

Main Type: Text-Image Content

Type 1: News Updates
Description: Brief text (usually under 500 words) with intuitive images, focusing on quick dissemination and instant interaction. Commonly used for hot topics, news, or personal updates.

Type 2: Deep Analysis
Description: Focus on deep content and rigorous logic, emphasizing opinion sharing, industry insights, or professional knowledge. Aims to deliver professional value.

Type 3: Lifestyle Guides
Description: High-quality images (often in 9-grid format) with text highlighting practicality and real experiences. Core concept is "documenting my life" to provide reference for others' consumption decisions.

Type 4: Product Reviews
Description: Professional, clear, and in-depth content providing decision-making basis through personal testing and detailed comparisons. Common in tech, appliances, etc.

Type 5: Community Discussion
Description: Deep discussions and creations around specific interests (like books, films, music, hobby groups). Strong community atmosphere with emphasis on authentic expression.

Type 6: Tutorial Guides
Description: Focus on clear step demonstration to help users "how to accomplish something". Requires logical structure and strong practicality.

Type 7: Collection Lists
Description: Compiling similar items or information systematically, such as product collections, book/movie lists, helping users quickly access systematic information.

Type 8: Opinion Posts
Description: Sharing personal insights and deep thoughts on social events or topics to spark discussion and resonance. Emphasizes logic and intellectual depth.

Type 9: Interactive Posts
Description: Actively inviting user participation through polls, topics, Q&A, etc. Aims to increase user engagement and interaction metrics.

Type 10: Story Content
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
        with open(idea_path, "w", encoding="utf-8") as f:
            json.dump(completion.choices[0].message.content, f, ensure_ascii=False, indent=2)
        print(completion.choices[0].message.content)
        print("Product Ideas generated.")
        return completion.choices[0].message.content



