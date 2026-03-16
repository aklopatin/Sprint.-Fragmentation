@echo off
REM Активация виртуального окружения (cmd)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Окружение активировано.
echo Обучение: python run_train.py
echo Инференс:  python run_inference.py image.jpg --checkpoint work_dirs\...\latest.pth
