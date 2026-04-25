@echo off
SETLOCAL EnableDelayedExpansion

echo ===========================================
echo   Starting NeuroSpectrum AI Server Suite
echo ===========================================

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

:: Install Python deps silently to avoid clutter unless it fails
echo [INFO] Verifying Python Requirements...
python -m pip install -r requirements.txt >nul 2>&1

:: Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    pause
    exit /b
)

:: Start Backend in a new window
echo [INFO] Launching FastAPI Backend (Port 8000)...
start "NeuroSpectrum Backend" cmd /k "title NeuroSpectrum-Backend && python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload"

:: Prepare Frontend
echo [INFO] Preparing React Frontend...
cd frontend

if not exist node_modules\ (
    echo [INFO] First time setup: Installing Node modules...
    call npm install
)

:: Start Frontend in a new window
echo [INFO] Launching React Frontend (Port 5173)...
start "NeuroSpectrum Frontend" cmd /k "title NeuroSpectrum-Frontend && npm run dev"

echo.
echo ===========================================
echo   All services launched in separate windows!
echo   You can safely close this window now.
echo ===========================================
timeout /t 5 >nul
