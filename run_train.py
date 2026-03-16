"""Запуск обучения: python run_train.py [--config ...] [--work-dir ...] [--amp]"""
import argparse
from mmengine.config import Config
from mmengine.runner import Runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/deeplabv3_fragmentation.py', help='Путь к конфигу')
    parser.add_argument('--work-dir', default='artifacts/deeplabv3_fragmentation', help='Папка для чекпоинтов и логов')
    parser.add_argument('--amp', action='store_true', help='Использовать mixed precision')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
