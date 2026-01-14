import os

# Fix OpenMP library conflict issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config.configurator import configs
from trainer.trainer import init_seed
from models.bulid_model import build_model
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer
from trainer.tuner import Tuner
import torch
import asyncio

def main():
    # First Step: Create data_handler
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    # Second Step: Create model
    model = build_model(data_handler).to(configs['device'])

    # Third Step: Create logger
    logger = Logger()

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)

    # Fifth Step: training
    if 'pretrain_path'  not in configs['train']:
        best_model = trainer.train(model)
    else:
        pretrain_path = configs['train']['pretrain_path']
        model.load_state_dict(torch.load(pretrain_path))
        best_model = trainer.load_model(model)

    # Dataset-specific post-processing
    dataset_name = configs['data']['name']

    if dataset_name == 'bilibili':
        import os
        # 检查是否跳过下载
        skip_download = os.getenv('SKIP_DOWNLOAD', 'false').lower() == 'true'

        if skip_download:
            print("\n" + "="*50)
            print("[INFO] 跳过视频下载 (SKIP_DOWNLOAD=true)")
            print("="*50)
        else:
            from data_utils.download import BilibiliDownloader
            downloader = BilibiliDownloader(best_model)
            asyncio.run(downloader.bilibili_test())

    elif dataset_name == 'redbook':
        import os
        # 检查是否跳过下载
        skip_download = os.getenv('SKIP_DOWNLOAD', 'false').lower() == 'true'

        if skip_download:
            print("\n" + "="*50)
            print("[INFO] 跳过小红书内容下载 (SKIP_DOWNLOAD=true)")
            print("="*50)
        else:
            from data_utils.download import RedbookDownloader
            downloader = RedbookDownloader(best_model)
            cookie = os.getenv('REDBOOK_COOKIE', '')
            downloader.add_cookies(cookie)
            asyncio.run(downloader.redbook_test())

    elif dataset_name == 'douban':
        import os
        # 检查是否跳过下载
        skip_download = os.getenv('SKIP_DOWNLOAD', 'false').lower() == 'true'

        if skip_download:
            print("\n" + "="*50)
            print("[INFO] 跳过豆瓣内容下载 (SKIP_DOWNLOAD=true)")
            print("="*50)
        else:
            from data_utils.download import DoubanDownloader
            downloader = DoubanDownloader(best_model)
            asyncio.run(downloader.douban_test())

    elif dataset_name == 'hupu':
        import os
        # 检查是否跳过下载
        skip_download = os.getenv('SKIP_DOWNLOAD', 'false').lower() == 'true'
        if skip_download:
            print("\n" + "="*50)
            print("[INFO] 跳过虎扑内容下载 (SKIP_DOWNLOAD=true)")
            print("="*50)
        else:
            from data_utils.download import HupuDownloader
            downloader = HupuDownloader(best_model)
            asyncio.run(downloader.hupu_test())



def tune():
    # First Step: Create data_handler
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    # Second Step: Create logger
    logger = Logger()

    # Third Step: Create tuner
    tuner = Tuner(logger)

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)

    # Fifth Step: Start grid search
    tuner.grid_search(data_handler, trainer)

if not configs['tune']['enable']:
    main()
else:
    tune()


