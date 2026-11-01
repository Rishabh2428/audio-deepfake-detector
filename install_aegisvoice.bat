@echo off
cd /d "%~dp0"
python --version
if errorlevel 1 (
  echo Python 3.12 or later is required.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe python -m venv .venv
if errorlevel 1 (
  echo Could not create .venv. Check that Python includes the venv and pip components.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo AegisVoice dependencies are installed. Run start_aegisvoice.bat next.
pause
exit /b 0
:failed
echo Dependency installation failed. Check internet access and the messages above.
pause
exit /b 1
