@echo off
chcp 65001 > nul
echo ====================================================
echo   DỰ ĐOÁN PHÂN LOẠI ẢNH CHÓ HAY MÈO
echo ====================================================
echo.
if "%~1"=="" (
    echo Đang kiểm tra với ảnh mẫu sample_cat.jpg...
    py -3.12 predict.py sample_cat.jpg
    echo.
    echo Đang kiểm tra với ảnh mẫu sample_dog.jpg...
    py -3.12 predict.py sample_dog.jpg
) else (
    py -3.12 predict.py "%~1"
)
pause
