# Активация виртуального окружения для mmsegmentation (PowerShell)
# Запуск: .\activate_env.ps1
Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1
Write-Host "Окружение активировано. Текущая папка: $(Get-Location)"
Write-Host "Обучение: python run_train.py"
Write-Host "Инференс:  python run_inference.py <image> --checkpoint work_dirs/.../latest.pth"
