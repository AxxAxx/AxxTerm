@echo off
echo === AxxTerm Build ===
echo.

python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    echo.
)

echo Building AxxTerm.exe...
python -m PyInstaller --onefile --windowed --name AxxTerm --clean AxxTerm_serial.py

echo.
if exist dist\AxxTerm.exe (
    echo Build successful! Executable: dist\AxxTerm.exe
) else (
    echo Build failed. Check the output above for errors.
)
pause
