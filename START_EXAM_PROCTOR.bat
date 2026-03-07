@echo off
setlocal
cd /d "%~dp0"

echo [INFO] Online CheatBuster launcher

set "VENV_DIR=.venv"
set "PY_EXE=%VENV_DIR%\\Scripts\\python.exe"

if not exist "%PY_EXE%" (
  echo [INFO] Creating virtual environment...
  py -3.11 -m venv "%VENV_DIR%" 2>nul || py -3 -m venv "%VENV_DIR%" 2>nul || python -m venv "%VENV_DIR%"
)

if not exist "%PY_EXE%" (
  echo [ERROR] Could not create venv. Install Python 3.11+ and retry.
  pause
  exit /b 1
)

echo [INFO] Installing requirements...
"%PY_EXE%" -m pip install --upgrade pip
"%PY_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Failed to install dependencies.
  pause
  exit /b 1
)

echo [INFO] Starting server at http://localhost:5001 ...
start "" "http://localhost:5001"
"%PY_EXE%" app.py

echo [INFO] Server stopped.
pause
exit /b 0
