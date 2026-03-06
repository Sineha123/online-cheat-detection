@echo off
setlocal

cd /d "%~dp0"

if not exist "app.py" (
  echo [ERROR] app.py not found in this folder.
  pause
  exit /b 1
)

set "VENV_PY=.venv311\Scripts\python.exe"
if exist "%VENV_PY%" goto deps

echo [INFO] Creating .venv311 ...
where python >nul 2>&1
if not errorlevel 1 (
  python -m venv .venv311
) else (
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3 -m venv .venv311
  ) else (
    where winget >nul 2>&1
    if not errorlevel 1 (
      echo [INFO] Python not found. Installing Python 3.11 via winget...
      winget install --id Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements
      where python >nul 2>&1
      if not errorlevel 1 (
        python -m venv .venv311
      ) else (
        where py >nul 2>&1
        if not errorlevel 1 (
          py -3 -m venv .venv311
        ) else (
          echo [ERROR] Python still not available after install.
          pause
          exit /b 1
        )
      )
    ) else (
      echo [ERROR] Python not found and winget not available.
      echo Install Python 3.11+ manually and re-run this file.
      start "" "https://www.python.org/downloads/"
      pause
      exit /b 1
    )
  )
)

if not exist "%VENV_PY%" (
  echo [ERROR] Could not create virtual environment.
  pause
  exit /b 1
)

:deps
if exist "requirements.txt" (
  if not exist ".venv311\.deps_installed" (
    echo [INFO] Installing requirements...
    "%VENV_PY%" -m pip install --upgrade pip
    rem Install all requirements except face_recognition first (avoid dlib source build on Windows)
    powershell -NoProfile -Command "(Get-Content 'requirements.txt') | Where-Object { $_ -notmatch '^\s*face_recognition\s*==' } | Set-Content '.venv311\requirements_no_face.txt'"
    "%VENV_PY%" -m pip install -r ".venv311\requirements_no_face.txt"
    if errorlevel 1 (
      echo [ERROR] base requirements install failed.
      pause
      exit /b 1
    )
    rem Install face_recognition without pulling dlib source dependency (dlib-bin already in requirements)
    "%VENV_PY%" -m pip install face_recognition==1.3.0 --no-deps
    if errorlevel 1 (
      echo [ERROR] face_recognition install failed.
      echo [HINT] Ensure internet is available and retry.
      pause
      exit /b 1
    )
    type nul > ".venv311\.deps_installed"
  )
)

echo [INFO] Starting app...
start "" "http://localhost:5000"
"%VENV_PY%" app.py

echo.
echo [INFO] App stopped.
pause
exit /b 0
