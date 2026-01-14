import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from tenacity import retry

SYSTEM_PROMPT_IMAGE_ANAYSIS = \
    """
    Your task is to analyze an image that a user has previously viewed from the following aspects:
    Briefly describe the content of the image, including both the pictures and any text, with each description.

    **Output Format**:
    Image Contents:
    """


class ImageAnalyst:
    def __init__(self, max_workers=3):
        self.image_client = OpenAI(
            api_key=os.getenv("IMAGE_API_KEY"),
            base_url=os.getenv("IMAGE_BASE_URL"),
        )
        self.max_workers = max_workers

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def prepare_user_content(self, image: str):
        # Get file extension to determine image type
        ext = os.path.splitext(image)[1].lower()
        # Map extensions to content types
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
        }
        if ext not in content_types:
            raise ValueError(f"Unsupported image format: {ext}")

        content_type = content_types[ext]
        base64_image = self.encode_image(image)

        user_content = [{
            "type": "image_url",
            "image_url": {"url": f"data:{content_type};base64,{base64_image}"}
        }]

        return user_content

    def call_api(self, user_content: list):
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=SYSTEM_PROMPT_IMAGE_ANAYSIS
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=user_content
            )
        ]
        try:
            print("Analyzing images...")
            completion = self.image_client.chat.completions.create(
                model=os.getenv("IMAGE_MODEL"),
                messages=messages
            )
            print("Image analysis completed.")
            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error during image analysis: {e}")
            raise e

    def analyze_single_image(self, idx, image_path):
        """分析单张图片的函数，用于并行处理"""
        try:
            user_content = self.prepare_user_content(image_path)
            image_analysis = self.call_api(user_content)
            return idx, image_analysis
        except Exception as e:
            print(f"Error analyzing image {image_path}: {e}")
            return idx, None

    def __call__(self, images: list, item_dir):
        if len(images)==0:
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

        # 获取已经分析过的image索引
        existing_image_indices = set()
        if "image" in all_analysis and isinstance(all_analysis["image"], dict):
            existing_image_indices = set(all_analysis["image"].keys())
            print(f"Found {len(existing_image_indices)} already analyzed images, will skip them.")

        # 过滤出有效的图片文件，并跳过已经分析过的
        valid_images = []
        for idx, image_path in enumerate(images):
            if image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                if str(idx) not in existing_image_indices:
                    valid_images.append((idx, image_path))
                else:
                    print(f"Skipping already analyzed image {idx}")

        if not valid_images:
            print("No new valid image files to analyze.")
            return

        # 初始化image分析结果存储
        image_results = {}
        successful_analyses = 0

        # 并行处理图片分析
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(self.analyze_single_image, idx, image_path): idx
                for idx, image_path in valid_images
            }

            # 收集结果
            for future in as_completed(future_to_idx):
                idx, analysis_result = future.result()
                if analysis_result is not None:
                    image_results[str(idx)] = analysis_result
                    successful_analyses += 1
                    print(f"Successfully analyzed image {idx}")

        # 只有在有成功的分析结果时才更新和保存文件
        if successful_analyses > 0:
            # 合并已有的和新的分析结果
            if "image" not in all_analysis:
                all_analysis["image"] = {}
            all_analysis["image"].update(image_results)

            with open(analysis_file_path, "w", encoding="utf-8") as f:
                json.dump(all_analysis, f, ensure_ascii=False, indent=2)
            print(f"Image analysis completed: {successful_analyses}/{len(valid_images)} new images analyzed successfully.")
            print(f"Total images in analysis: {len(all_analysis['image'])}")
        else:
            print("No new images were successfully analyzed. Analysis file not updated.")
