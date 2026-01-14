import os
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from tenacity import retry

SYSTEM_PROMPT_USER_PROFILE_GENERATOR = \
"""
Your task is to generate a comprehensive user profile based on the previous analysis of notes the user has viewed, following these requirements:
1.List the user's top 5 preferences, from highest to lowest. Each preference should be described with a brief phrase, no more than 200 words.
2.After each preference, provide the reason for it in parentheses, such as previously viewed items, or prior analyses, no more than 200 words.
3.Historical items are those that the user has previously interacted with and have a high confidence level, while recommended items are system-generated suggestions with lower confidence, requiring careful evaluation of their reliability.

**Output Format**:
Ordering by user preference level, from highest to lowest:
1. Preference 1:
   Reason:
2. Preference 2:
   Reason:
...
5. Preference 5:
   Reason:
"""

USER_PROMPT_PRIOR_ANALYSIS = \
"""
**Previous Note Analysis**:
{note_analysis}
"""

class UserProfileGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("CHAT_API_KEY"),
            base_url=os.getenv("CHAT_BASE_URL"),
        )

    @retry
    def __call__(self, item_profiles: str):
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=SYSTEM_PROMPT_USER_PROFILE_GENERATOR
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=USER_PROMPT_PRIOR_ANALYSIS.format(
                    note_analysis=item_profiles
                )
            )
        ]
        try:
            print("Generating user profile...")
            completion = self.client.chat.completions.create(
                model=os.getenv("CHAT_MODEL"),
                messages=messages
            )
            print("User profile generated.")
            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error during image analysis: {e}")
            raise e

