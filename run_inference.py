"""Инференс по изображению (чекпоинт best_mDice по умолчанию)."""
import argparse
from pathlib import Path
from mmseg.apis import inference_model, init_model, show_result_pyplot

from scripts.training_logging import get_best_mdice_checkpoint

DEFAULT_WORK_DIR = Path("artifacts/deeplabv3_fragmentation")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Путь к изображению')
    parser.add_argument('--config', default='configs/deeplabv3_fragmentation.py', help='Конфиг модели')
    parser.add_argument('--checkpoint', default=None, help='Путь к чекпоинту .pth (по умолчанию: best_mDice в --work-dir)')
    parser.add_argument('--work-dir', type=Path, default=DEFAULT_WORK_DIR, help='Папка с чекпоинтами (если --checkpoint не задан)')
    parser.add_argument('--out', default='result.png', help='Файл для сохранения визуализации')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 или cpu')
    parser.add_argument('--opacity', type=float, default=0.5, help='Прозрачность маски (0-1)')
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not ckpt:
        best = get_best_mdice_checkpoint(args.work_dir)
        if best is None:
            raise SystemExit(
                f"Чекпоинт не найден: в {args.work_dir} нет best_mDice_iter_*.pth. "
                "Укажите --checkpoint путь/к/файлу.pth или сначала запустите обучение."
            )
        ckpt = str(best)
    else:
        ckpt = str(Path(ckpt).resolve())

    model = init_model(args.config, ckpt, device=args.device)
    result = inference_model(model, args.image)
    show_result_pyplot(model, args.image, result, show=False, out_file=args.out, opacity=args.opacity)
    print('Сохранено:', args.out)

if __name__ == '__main__':
    main()
