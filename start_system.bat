@echo off
title Trading System Launcher
color 0A

:: Switch to repository directory
cd /d "C:\Users\satya\OneDrive\Documents\GitHub\trading-system"

echo =====================================================================
echo           TRADING SYSTEM - TECHNICAL ADVISOR LAUNCHER
echo =====================================================================
echo.

if not exist ".\.venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found in .\.venv
    echo Please make sure the folder exists.
    pause
    exit /b 1
)

echo [*] Launching FastAPI Backend on http://127.0.0.1:8000 ...
start "FastAPI Backend (Port 8000)" cmd /k "cd /d "C:\Users\satya\OneDrive\Documents\GitHub\trading-system" && .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo [*] Launching React Dashboard on http://localhost:5173 ...
start "React Dashboard (Port 5173)" cmd /k "cd /d "C:\Users\satya\OneDrive\Documents\GitHub\trading-system\frontend" && npm run dev -- --host 0.0.0.0"

echo.
echo [*] Waiting 4 seconds for servers to initialize...
ping 127.0.0.1 -n 5 >nul

echo [*] Opening Dashboard in default web browser...
start http://localhost:5173/

echo.
echo =====================================================================
echo                    SERVICES STARTED SUCCESSFULLY!
echo =====================================================================
echo  - Frontend Dashboard : http://localhost:5173/
echo  - Backend API Docs   : http://127.0.0.1:8000/docs
echo.
echo  NEXT STEP:
echo  1. In the browser, log into Zerodha.
echo  2. Click the 'Technical Advisor' tab to scan any NSE stock!
echo.
echo  (You can minimize this window. The servers run in their own windows.)
echo =====================================================================
echo.
pause
