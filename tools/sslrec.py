import os
import re
import subprocess
import sys

def run_sslrec():
    current_dir = os.getcwd()
    sslrec_dir = os.path.join(current_dir, 'SSLRec')

    if not os.path.exists(sslrec_dir):
        print(f"错误: SSLRec目录不存在: {sslrec_dir}")
        return False

    dataset = os.getenv("DATASET")  # Default to bilibili if not set
    model = os.getenv("MODEL", "SGL")  # Default to SGL if not set
    cmd = [sys.executable, 'main.py', '--model', model, '--dataset', dataset]

    try:
        print(f"🚀 启动SSLRec: {' '.join(cmd)}")
        print(f"📁 工作目录: {sslrec_dir}")
        print("-" * 50)

        process = subprocess.Popen(
            cmd,
            cwd=sslrec_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=0,  # 无缓冲，实时输出
            encoding='utf-8',
            errors='replace'
        )

        # 用于跟踪当前显示的进度条类型
        current_progress_type = None

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break

            if output:
                line = output.rstrip()

                # 检测是否为进度条输出
                progress_match = re.search(r'(Training Recommender|Epoch|Testing)', line)
                if progress_match and ('|' in line or '%' in line):
                    progress_type = progress_match.group(1)

                    # 如果进度条类型改变，先换行
                    if current_progress_type and current_progress_type != progress_type:
                        print()  # 换行

                    current_progress_type = progress_type

                    # 在同一行更新进度条
                    print(f"\r{line}", end='', flush=True)
                else:
                    # 非进度条输出，正常打印
                    if current_progress_type:
                        print()  # 进度条结束后换行
                        current_progress_type = None
                    print(line, flush=True)  # 添加 flush=True 确保实时输出

        # 确保最后有换行
        if current_progress_type:
            print()

        return_code = process.poll()

        print("-" * 50)
        if return_code == 0:
            print("✅ SSLRec 执行成功")
        else:
            print(f"❌ SSLRec 执行失败，退出代码: {return_code}")

        return return_code == 0

    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
        process.terminate()
        process.wait()
        return False
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False
