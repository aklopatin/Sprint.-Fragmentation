# MMSegmentation: обучение на train_dataset_for_students

## Что не в репозитории (минимальный размер)

В репозитории только код и конфиги. Не хранятся: датасет `train_dataset_for_students/`, виртуальное окружение `.venv/`, чекпоинты и логи `artifacts/`, видео `sourcevideo.mp4` / `output_segmented.mp4`. После клонирования: создать `.venv` и установить зависимости (`pip install -r requirements.txt`), положить датасет в `train_dataset_for_students/` по структуре ниже.

## Окружение (уже подготовлено)

- **Виртуальное окружение:** `.venv`
- **Активация (PowerShell):** `.\activate_env.ps1`
- **Активация (cmd):** `activate_env.bat`

После активации все команды ниже выполнять из корня проекта.

## Датасет

- **Путь:** `train_dataset_for_students/`
- **Структура:** `img/{train,val,test}/` — изображения `.jpg`, `labels/{train,val,test}/` — маски `.png` (те же имена).
- **Классы:** 3 (метки 0, 1, 2). Проверка: `python scripts/check_num_classes.py`.

## Конфиги

- `configs/fragmentation_dataset.py` — описание датасета и пайплайны.
- `configs/deeplabv3_fragmentation.py` — модель DeepLabV3 (ResNet50) под этот датасет, обучение на GPU.
- `configs/pspnet_fragmentation.py` — альтернативный конфиг PSPNet (ResNet50).

При другом числе классов измените `num_classes` в `fragmentation_dataset.py` и в `decode_head`/`auxiliary_head` выбранного модельного конфига.

## Обучение (GPU)

```bash
python run_train.py
```

Опции: `--config`, `--work-dir`, `--amp` (mixed precision).

Чекпоинты, логи и артефакты: `artifacts/deeplabv3_fragmentation/` (или каталог из `--work-dir`).
Лучший чекпоинт сохраняется автоматически по `mDice` (best model).

## Инференс (GPU)

Используется чекпоинт **best_mDice** (файл `best_mDice_iter_*.pth`), не last:

```bash
python run_inference.py путь/к/картинке.jpg --out result.png
```

(по умолчанию берётся `artifacts/deeplabv3_fragmentation`; при необходимости укажите `--checkpoint путь/к/best_mDice_iter_4000.pth` или `--work-dir папка`)

## Замечание по mmcv на Windows

Установлен **mmcv-lite** (без CUDA-операторов), чтобы обойтись без сборки mmcv под Windows. Для обучения и инференса на GPU этого достаточно; часть оптимизаций может быть медленнее, чем с полным mmcv. Полный mmcv с CUDA под Windows можно собрать из исходников или использовать Linux/WSL.
