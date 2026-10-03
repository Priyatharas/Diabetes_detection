@echo off
title Diabetes Prediction AI Application Launcher
echo ========================================================
echo   Starting Diabetes Prediction AI Application
echo ========================================================
echo.

echo [1/2] Starting Flask Backend (Port 5000)...
start "Diabetes Backend (Flask)" cmd /k "cd /d %~dp0backend && python app.py"

echo [2/2] Starting React + Vite Frontend (Port 5173)...
start "Diabetes Frontend (Vite)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================================
echo Both servers are starting up!
echo   - Backend API: http://127.0.0.1:5000
echo   - Frontend UI: http://localhost:5173
echo ========================================================
echo Opening http://localhost:5173 in your default browser...
timeout /t 3 >nul
start http://localhost:5173
