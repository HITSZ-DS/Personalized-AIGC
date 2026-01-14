import json
import os
from concurrent.futures import as_completed, ThreadPoolExecutor

from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

SYSTEM_PROMPT_TEXT_ANALYSIS = \
"""
Your task is to analyze a text document that a user has previously provided from the following aspects:
Briefly describe the content of the text, including both the main themes and any specific details, with each description.

Output Format:
Text Contents:
"""

class TextAnalyst:
    def __init__(self, max_workers=3):
        self.client = OpenAI(
            api_key=os.getenv("CHAT_API_KEY"),
            base_url=os.getenv("CHAT_BASE_URL")
        )
        self.max_workers = max_workers

    def call_api(self, user_content: str):
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=SYSTEM_PROMPT_TEXT_ANALYSIS
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=user_content
            )
        ]
        try:
            print("Analyzing text...")
            completion = self.client.chat.completions.create(
                model=os.getenv("CHAT_MODEL"),
                messages=messages
            )
            print("Text analysis completed.")
            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error during text analysis: {e}")
            raise e

    def analyze_single_text(self, idx, text_path):
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                user_content = f.read()
            text_analysis = self.call_api(user_content)
            return idx, text_analysis
        except Exception as e:
            print(f"Error analyzing text {text_path}: {e}")
            return idx, None

    def __call__(self, text_list: list, item_dir):
        if len(text_list)==0:
            return

        # 加载现有的analysis.json（如果存在）
        analysis_file_path = os.path.join(item_dir, "analysis.json")
        if os.path.exists(analysis_file_path):
            try:
                with open(analysis_file_path, "r", encoding="utf-8") as f:
                    all_analysis = json.load(f)
            except Exception as e:
                print(f"Error loading existing analysis.json: {e}")
                all_analysis = {}
        else:
            all_analysis = {}

        # 获取已经分析过的text索引
        existing_text_indices = set()
        if "text" in all_analysis and isinstance(all_analysis["text"], dict):
            existing_text_indices = set(all_analysis["text"].keys())
            print(f"Found {len(existing_text_indices)} already analyzed texts, will skip them.")

        # 过滤出有效的文本文件，并跳过已经分析过的
        valid_text = []
        for idx, text_path in enumerate(text_list):
            if text_path.lower().endswith(('.txt')):
                if str(idx) not in existing_text_indices:
                    valid_text.append((idx, text_path))
                else:
                    print(f"Skipping already analyzed text {idx}")

        if not valid_text:
            print("No new valid text files to analyze.")
            return

        # 初始化text分析结果存储
        text_results = {}
        successful_analyses = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.analyze_single_text, idx, text_path): idx
                for idx, text_path in valid_text
            }

            for future in as_completed(future_to_idx):
                idx, analysis_result = future.result()
                if analysis_result is not None:
                    text_results[str(idx)] = analysis_result
                    successful_analyses += 1
                    print(f"Successfully analyzed text {idx}")

        # 只有在有成功的分析结果时才更新和保存文件
        if successful_analyses > 0:
            # 合并已有的和新的分析结果
            if "text" not in all_analysis:
                all_analysis["text"] = {}
            all_analysis["text"].update(text_results)

            with open(analysis_file_path, "w", encoding="utf-8") as f:
                json.dump(all_analysis, f, ensure_ascii=False, indent=2)
            print(f"Text analysis completed: {successful_analyses}/{len(valid_text)} new texts analyzed successfully.")
            print(f"Total texts in analysis: {len(all_analysis['text'])}")
        else:
            print("No new texts were successfully analyzed. Analysis file not updated.")
