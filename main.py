import os.path
from tools.product import product
from tools.analyze import analyze
from tools.enhanced_analyze import enhanced_analyze
from tools.enhanced_analyze_round10 import enhanced_analyze_round10
from tools.enhanced_realtime_analyze import enhanced_realtime_analyze
from tools.sslrec import run_sslrec
from dotenv import load_dotenv
from tools.bilibili_realtime_download import BilibiliRealTime
from tools.redbook_realtime import RedBookRealtime
from tools.realtime_analyze import RealtimeAnalyzer

if __name__ == "__main__":
    load_dotenv()
    dataset = os.getenv("DATASET")  # Default to bilibili if not set

    print(f"🎯 使用数据集: {dataset}")

    # Run SSL recommendation model
    # run_sslrec()      # 推荐商品+下载三集交互的数据
 
    # Analyze data
    # First batch: analyze first 250 users
    #analyze(max_users=1000)
    
    # Second batch: analyze next 200 users (skip first 250)
    #analyze(max_users=200, skip_users=250)

    # Enhanced analysis with hit analysis and self-reflection (now with concurrent processing)
    # First batch: enhanced_analyze first 250 users
    enhanced_analyze(max_concurrent_users=40)
    
    # Second batch: enhanced_analyze next 200 users (skip first 250)
    # enhanced_analyze(max_concurrent_users=40, max_users=200, skip_users=250)
    
    # enhanced_analyze_round10(max_concurrent_users=40, max_users=1000)
    # Second batch: enhanced_analyze_round10 next 200 users (skip first 250)
    #enhanced_analyze_round10(max_concurrent_users=40, max_users=200, skip_users=250)


    # Product generation
    # product()

    # Real-time data collection and download
    # 
    # 小红书实时爬取（新版）- 快速启动（不需要交互式输入）：
    # if dataset == "redbook":
    #     redbook_realtime = RedBookRealtimeNew()  # 使用脚本中配置的Cookie
    #     redbook_realtime.process_all_users(
    #         save_data=True, 
    #         download_media=True,
    #         max_users=200,  # 只处理前200个用户
    #         parallel=3      # 3个并发
    #     )
    # 
    # if dataset == "bilibili":
    #     bilibili_realtime = BilibiliRealTime()
    #     # Process all users automatically (no need to input user_id)
    #     print("🚀 开始自动处理所有用户的实时数据...")
    #     bilibili_realtime.process_all_users(save_data=True, download_videos=True)
    # elif dataset == "redbook":
    #     # Ask for cookies (optional, will use default from script)
    #     cookies = input("请输入小红书cookies (回车使用默认): ").strip()
    #     
    #     # Ask for parallel and max_users settings
    #     parallel_input = input("并发数 (默认3，回车跳过): ").strip()
    #     parallel = int(parallel_input) if parallel_input.isdigit() else 3
    #     
    #     max_users_input = input("用户上限 (默认所有，回车跳过): ").strip()
    #     max_users = int(max_users_input) if max_users_input.isdigit() else None
    #     
    #     redbook_realtime = RedBookRealtimeNew(cookies if cookies else None)
    #     print(f"🚀 开始自动处理所有用户的实时数据 (并发: {parallel}, 用户上限: {max_users or '所有'})...")
    #     redbook_realtime.process_all_users(
    #         save_data=True, 
    #         download_media=True, 
    #         max_users=max_users,
    #         parallel=parallel
    #     )

    # Realtime analysis
    # realtime_analyze = RealtimeAnalyzer(folder_max_workers=4)
    # realtime_analyze()

    # Enhanced realtime analysis
    # enhanced_realtime_analyze(max_concurrent_users=4)






