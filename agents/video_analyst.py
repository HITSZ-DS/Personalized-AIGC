import base64
import json
import math
import os
import shutil
import warnings
import time
import uuid
import http.client
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from tenacity import retry
from moviepy.editor import VideoFileClip
import tempfile

# 抑制 MoviePy 的特定警告
warnings.filterwarnings("ignore", message=".*bytes wanted but.*bytes read.*")
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")

SYSTEM_PROMPT_SINGLE_VIDEO_ANALYSIS = \
    """
    Your task is to analyze a video that a user has previously viewed from the following aspects:
    Briefly describe the content of the video, including both the visual elements (scenes, actions, people, etc.) and any audio/narrative components (dialogue, voiceover, music, on-screen text), with description not exceeding 200 words.

    **Output Format**:
    Video Contents:
    """

SYSTEM_PROMPT_ALL_SEGMENT_ANALYSIS = \
    """
    Your task is to summarize the user's explicit (directly observable) and implicit (underlying, inferred) preferences based on the analysis of video segments provided below.
    These segments are the beginning or ending parts (each 8 seconds) of a longer video that the user has previously viewed. Each segment has already been analyzed individually, and your task is to synthesize these analyses to provide a comprehensive understanding of the user's preferences from the following aspects:
    1. Briefly summarize the content of the longer video based on the analyses of its beginning and ending segments, with summary not exceeding 300 words.
    2. Analyze the connections and common features between the beginning and ending segments from multiple perspectives (e.g., thematic, stylistic, narrative, genre, aesthetic, or technical aspects).

    **Output Format**:
    1. Video Contents:
    2. Common Features:
    """


class VideoAnalyst:
    def __init__(self, max_workers=2, video_max_workers=None):
        self.video_client = OpenAI(
            api_key=os.getenv("VIDEO_API_KEY"),
            base_url=os.getenv("VIDEO_BASE_URL"),
        )
        self.chat_client = OpenAI(
            api_key=os.getenv("CHAT_API_KEY"),
            base_url=os.getenv("CHAT_BASE_URL"),
        )
        # 新增API配置
        self.api_key = os.getenv("VIDEO_API_KEY")
        self.auth_token = os.getenv("VIDEO_API_KEY")
        self.max_workers = max_workers  # 单个视频内片段分析的并发数
        self.video_max_workers = video_max_workers or max_workers * 2  # 多个视频之间的并发数
        self.print_lock = Lock()
        # 设置logger
        self.logger = logging.getLogger(__name__)

    def safe_print(self, message):
        """线程安全的打印函数"""
        with self.print_lock:
            print(message)

    def encode_video(self, video_path):
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode("utf-8")

    def create_segment_safe(self, video_path, start_time, end_time, output_path, segment_type):
        """安全地创建视频片段，避免线程问题和文件访问冲突"""
        video = None
        segment = None
        max_retries = 3
        temp_video_path = None

        for attempt in range(max_retries):
            try:
                # 检查视频路径是否包含非ASCII字符
                # 如果包含，复制到临时位置以避免FFmpeg编码问题
                try:
                    video_path.encode('ascii')
                    working_video_path = video_path
                except UnicodeEncodeError:
                    # 路径包含非ASCII字符，需要复制到临时位置
                    if temp_video_path is None:  # 只在第一次尝试时复制
                        temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_video_{uuid.uuid4().hex}.mp4")
                        self.safe_print(f"Video path contains non-ASCII characters, copying to: {temp_video_path}")
                        shutil.copy2(video_path, temp_video_path)
                    working_video_path = temp_video_path

                # 为每个线程创建新的视频对象
                video = VideoFileClip(working_video_path)
                duration = video.duration

                # 确保时间范围有效，避免接近视频末尾的问题帧
                # 为视频末尾留出1秒的缓冲区以避免FFmpeg错误
                safe_duration = max(0, duration - 1.0)
                start_time = max(0, min(start_time, safe_duration))
                end_time = max(start_time + 0.1, min(end_time, safe_duration))  # 确保至少0.1秒的片段

                if end_time <= start_time:
                    raise ValueError(f"Invalid time range: {start_time} to {end_time}")

                self.safe_print(f"Creating {segment_type} segment: {start_time:.2f}s to {end_time:.2f}s (attempt {attempt + 1})")

                # 创建片段
                segment = video.subclip(start_time, end_time)

                # 创建唯一的临时目录来避免文件名冲突
                unique_temp_dir = tempfile.mkdtemp(prefix=f"moviepy_{segment_type}_{uuid.uuid4().hex[:8]}_")

                # 使用上下文管理器抑制警告
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.filterwarnings("ignore", message=".*bytes wanted but.*bytes read.*")

                    # 写入文件，使用唯一的临时目录
                    # 使用更宽松的编码参数来提高兼容性
                    segment.write_videofile(
                        output_path,
                        codec="libx264",
                        audio_codec="aac",
                        temp_audiofile=os.path.join(unique_temp_dir, f"temp_audio_{uuid.uuid4().hex[:8]}.wav"),
                        remove_temp=True,
                        logger=None,  # 禁用日志输出
                        preset='ultrafast',  # 使用最快的编码预设
                        threads=4,  # 限制线程数以减少资源竞争
                        fps=24,  # 统一帧率
                        bitrate='2000k'  # 设置适中的比特率
                    )

                # 清理临时目录
                try:
                    shutil.rmtree(unique_temp_dir)
                except:
                    pass

                self.safe_print(f"Successfully created {segment_type} segment: {output_path}")
                return output_path

            except Exception as e:
                error_msg = str(e)
                self.safe_print(f"Error creating {segment_type} segment (attempt {attempt + 1}): {error_msg}")

                # 清理可能创建的文件
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass

                # 处理各种FFmpeg和文件访问错误
                retry_errors = [
                    "WinError 32", "another program",
                    "Resource temporarily unavailable",
                    "Error opening input file",
                    "Invalid argument",
                    "Broken pipe",
                    "malloc"
                ]

                should_retry = any(err_pattern in error_msg for err_pattern in retry_errors)

                if should_retry and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1.0  # 增加等待时间到1秒倍数
                    self.safe_print(f"Encountered recoverable error, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                # 如果是最后一次尝试，抛出异常
                if attempt == max_retries - 1:
                    raise e

            finally:
                # 确保资源被正确释放
                if segment is not None:
                    try:
                        segment.close()
                    except:
                        pass
                if video is not None:
                    try:
                        video.close()
                    except:
                        pass
                # 清理临时视频文件
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                        self.safe_print(f"Cleaned up temporary video file: {temp_video_path}")
                    except:
                        pass
                # 重置变量以便下次重试
                segment = None
                video = None

    def split_video(self, video_path, segment_duration=5):
        """并行分割视频为开始和结束片段"""
        # 首先获取视频信息
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            video = VideoFileClip(video_path)
            duration = video.duration
            video.close()

        cur_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # 使用唯一的临时目录，只使用ASCII字符避免编码问题
        unique_id = uuid.uuid4().hex[:8]
        temp_dir = tempfile.mkdtemp(prefix=f"video_seg_{unique_id}_")
        self.safe_print(f"Created temporary directory: {temp_dir}")

        segments = []
        segment_tasks = []

        # 为视频末尾留出安全缓冲区
        safe_duration = max(0, duration - 1.0)  # 末尾留出1秒缓冲

        # 准备分割任务 - 只分析开始和结束部分
        start_path = os.path.join(temp_dir, f"segment_start.mp4")
        segment_tasks.append(("start", 0, min(segment_duration, safe_duration), start_path))

        # 只有当视频长度超过segment_duration时才创建end片段
        if safe_duration > segment_duration:
            end_start = max(0, safe_duration - segment_duration)
            end_path = os.path.join(temp_dir, f"segment_end.mp4")
            segment_tasks.append(("end", end_start, safe_duration, end_path))

        # 并行处理片段创建
        segments = []
        with ThreadPoolExecutor(max_workers=len(segment_tasks)) as executor:
            # 提交所有片段创建任务
            future_to_task = {
                executor.submit(
                    self.create_segment_safe,
                    video_path, start_time, end_time, output_path, segment_type
                ): (segment_type, output_path)
                for segment_type, start_time, end_time, output_path in segment_tasks
            }

            # 收集结果
            for future in as_completed(future_to_task):
                segment_type, output_path = future_to_task[future]
                try:
                    segment_path = future.result()
                    segments.append(segment_path)
                    self.safe_print(f"Successfully created {segment_type} segment in parallel")
                except Exception as e:
                    self.safe_print(f"Failed to create {segment_type} segment: {e}")
                    continue

        # 按照原始顺序排序片段
        segment_order = {"start": 0, "end": 1}
        segments.sort(key=lambda x: segment_order.get(
            os.path.basename(x).split('_')[1].split('.')[0], 2
        ))

        return segments, temp_dir

    # def analyze_single_segment(self, segment_path):
    #     """分析单个视频片段"""
    #     try:
    #         if not os.path.exists(segment_path):
    #             raise FileNotFoundError(f"Segment file not found: {segment_path}")

    #         segment_content = []
    #         base64_video = self.encode_video(segment_path)
    #         segment_content.append({
    #             "type": "video_url",
    #             "video_url": {"url": f"data:video/mp4;base64,{base64_video}"}
    #         })

    #         self.safe_print(f"Analyzing segment: {os.path.basename(segment_path)}")
    #         result = self.call_single_api(segment_content)
    #         self.safe_print(f"Completed analysis for: {os.path.basename(segment_path)}")
    #         return result

    #     except Exception as e:
    #         self.safe_print(f"Error analyzing segment {segment_path}: {e}")
    #         raise e

    # def call_single_api(self, user_content: list):
    #     messages = [
    #         ChatCompletionSystemMessageParam(
    #             role="system",
    #             content=SYSTEM_PROMPT_SINGLE_VIDEO_ANALYSIS
    #         ),
    #         ChatCompletionUserMessageParam(
    #             role="user",
    #             content=user_content
    #         )
    #     ]
    #     try:
    #         completion = self.video_client.chat.completions.create(
    #             model=os.getenv("VIDEO_MODEL"),
    #             messages=messages
    #         )
    #         return completion.choices[0].message.content

    #     except Exception as e:
    #         self.safe_print(f"Error during segment analysis: {e}")
    #         raise e

    def analyze_single_segment(self, segment_path):
        """分析单个视频片段 - 使用新的API调用方式"""
        try:
            if not os.path.exists(segment_path):
                raise FileNotFoundError(f"Segment file not found: {segment_path}")

            self.safe_print(f"Analyzing segment: {os.path.basename(segment_path)}")

            # 编码视频为base64
            video_base64 = self._encode_video_to_base64(segment_path)

            # 调用API
            result = self.call_single_api(video_base64)

            self.safe_print(f"Completed analysis for: {os.path.basename(segment_path)}")
            return result

        except Exception as e:
            self.safe_print(f"Error analyzing segment {segment_path}: {e}")
            raise e

    def _encode_video_to_base64(self, video_path: str) -> str:
        """Convert video file to base64 string"""
        try:
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
                base64_encoded = base64.b64encode(video_bytes).decode('utf-8')
                return base64_encoded
        except Exception as e:
            self.logger.error(f"Failed to encode video {video_path}: {e}")
            raise

    def call_single_api(self, video_base64: str):
        """使用http.client调用API进行视频分析"""
        try:
            # 准备API请求
            payload = json.dumps({
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "video/mp4",
                                    "data": video_base64
                                }
                            },
                            {
                                "text": SYSTEM_PROMPT_SINGLE_VIDEO_ANALYSIS
                            }
                        ]
                    }
                ]
            })

            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }

            # 发送API请求
            self.logger.info("Sending request to multimodal model...")
            conn = http.client.HTTPSConnection("yunwu.ai")
            video_model = os.getenv("VIDEO_MODEL")
            conn.request("POST", f"/v1beta/models/{video_model}:generateContent?key={self.api_key}", payload, headers)
            res = conn.getresponse()
            data = res.read()

            response_text = data.decode("utf-8")
            self.logger.info(f"API response received: {response_text[:200]}...")

            # 解析响应
            try:
                response_json = json.loads(response_text)

                # 提取实际响应内容
                if 'candidates' in response_json and len(response_json['candidates']) > 0:
                    content = response_json['candidates'][0]['content']['parts'][0]['text']
                    return content
                else:
                    raise ValueError("No candidates found in API response")

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.error(f"Failed to parse API response: {e}")
                self.logger.error(f"Raw response: {response_text}")
                raise

        except Exception as e:
            self.safe_print(f"Error during segment analysis: {e}")
            raise e

    def call_all_api(self, user_content: list):
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=SYSTEM_PROMPT_ALL_SEGMENT_ANALYSIS
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=user_content
            )
        ]
        try:
            self.safe_print("Analyzing video...")
            completion = self.chat_client.chat.completions.create(
                model=os.getenv("CHAT_MODEL"),
                messages=messages
            )
            self.safe_print("All segments analysis completed.")
            return completion.choices[0].message.content

        except Exception as e:
            self.safe_print(f"Error during video analysis: {e}")
            raise e

    def cleanup_temp_files(self, temp_dir):
        """清理临时文件"""
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                self.safe_print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                self.safe_print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")

    def process_single_video(self, idx, video_path):
        """处理单个视频的函数，用于并行处理"""
        try:
            self.safe_print(f"Starting to process video {idx}: {os.path.basename(video_path)}")

            video_content = []
            temp_dir = None

            # 检查视频时长
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                video = VideoFileClip(video_path)
                duration = video.duration
                self.safe_print(f"Video {idx} duration: {duration} seconds")
                video.close()

            # 分割视频
            if duration > 8:
                segments, temp_dir = self.split_video(video_path)

                if not segments:
                    raise RuntimeError("Failed to create any video segments")

                # 并行分析所有片段
                analyses = []
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_segment = {
                        executor.submit(self.analyze_single_segment, segment): segment
                        for segment in segments
                    }

                    for future in as_completed(future_to_segment):
                        segment = future_to_segment[future]
                        try:
                            analysis = future.result()
                            analyses.append((segment, analysis))
                            self.safe_print(f"Analysis result for {os.path.basename(segment)}:")
                            self.safe_print(analysis)
                        except Exception as e:
                            self.safe_print(f"Error analyzing {segment}: {e}")
                            continue

                if not analyses:
                    raise RuntimeError("Failed to analyze any video segments")

                # 按照片段顺序排序分析结果
                segment_order = {"start": 0, "end": 1}
                analyses.sort(key=lambda x: segment_order.get(
                    os.path.basename(x[0]).split('_')[1].split('.')[0], 2
                ))

                # 准备最终分析的输入
                for segment, analysis in analyses:
                    video_content.append({
                        "type": "text",
                        "text": analysis
                    })

            else:
                # 视频较短，直接分析
                base64_video = self._encode_video_to_base64(video_path)
                single = self.call_single_api(base64_video)
                self.safe_print(f"Single video {idx} analysis:")
                self.safe_print(single)
                video_content = [{
                    "type": "text",
                    "text": single
                }]

            # 进行最终的综合分析
            analysis = self.call_all_api(video_content)
            self.safe_print(f"Completed processing video {idx}: {os.path.basename(video_path)}")

            # 清理临时文件
            if temp_dir:
                self.cleanup_temp_files(temp_dir)

            return idx, analysis

        except Exception as e:
            self.safe_print(f"Error processing video {idx} ({video_path}): {e}")
            # 确保清理临时文件
            if 'temp_dir' in locals() and temp_dir:
                self.cleanup_temp_files(temp_dir)
            return idx, None

    def __call__(self, videos: list, item_dir):
        if len(videos)==0:
            return

        # 加载现有的analysis.json（如果存在）
        analysis_file_path = os.path.join(item_dir, "analysis.json")
        if os.path.exists(analysis_file_path):
            try:
                with open(analysis_file_path, "r", encoding="utf-8") as f:
                    all_analysis = json.load(f)
            except Exception as e:
                self.safe_print(f"Error loading existing analysis.json: {e}")
                all_analysis = {}
        else:
            all_analysis = {}

        # 获取已经分析过的video索引
        existing_video_indices = set()
        if "video" in all_analysis and isinstance(all_analysis["video"], dict):
            existing_video_indices = set(all_analysis["video"].keys())
            self.safe_print(f"Found {len(existing_video_indices)} already analyzed videos, will skip them.")

        # 过滤出有效的视频文件，并跳过已经分析过的
        valid_videos = []
        for idx, video_path in enumerate(videos):
            if video_path.lower().endswith('.mp4'):
                if str(idx) not in existing_video_indices:
                    valid_videos.append((idx, video_path))
                else:
                    self.safe_print(f"Skipping already analyzed video {idx}")

        if not valid_videos:
            self.safe_print("No new valid MP4 videos to analyze.")
            return

        # 初始化video分析结果存储
        video_results = {}
        successful_analyses = 0

        # 并行处理所有视频 - 使用专门的视频级别并发数
        video_max_workers = self.video_max_workers  # 视频级别的并发数
        print(f"视频级别的并发数: {video_max_workers}")
        with ThreadPoolExecutor(max_workers=video_max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(self.process_single_video, idx, video_path): idx
                for idx, video_path in valid_videos
            }

            # 收集结果
            for future in as_completed(future_to_idx):
                idx, analysis_result = future.result()
                if analysis_result is not None:
                    video_results[str(idx)] = analysis_result
                    successful_analyses += 1
                    self.safe_print(f"Successfully analyzed video {idx}")

        # 只有在有成功的分析结果时才更新和保存文件
        if successful_analyses > 0:
            # 合并已有的和新的分析结果
            if "video" not in all_analysis:
                all_analysis["video"] = {}
            all_analysis["video"].update(video_results)

            print('开始保存analysis.json')
            with open(analysis_file_path, "w", encoding="utf-8") as f:
                json.dump(all_analysis, f, ensure_ascii=False, indent=2)
            print('保存结束')
            self.safe_print(f"Video analysis completed: {successful_analyses}/{len(valid_videos)} new videos analyzed successfully.")
            self.safe_print(f"Total videos in analysis: {len(all_analysis['video'])}")
        else:
            self.safe_print("No new videos were successfully analyzed. Analysis file not updated.")
