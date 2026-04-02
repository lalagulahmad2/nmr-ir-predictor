@echo off
title NMR ^& IR Spectrum Predictor
echo ============================================
echo   NMR ^& IR Spectrum Predictor
echo ============================================
echo.

REM Try 'python' first, then 'py' (Python Launcher for Windows)
where python >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=python
    goto :run
)
where py >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=py
    goto :run
)

echo ERROR: Python not found on PATH.
echo Please install Python 3.10+ from https://python.org
echo Make sure to check "Add Python to PATH" during install.
echo.
pause
exit /b 1

:run
echo Starting server...
echo Open your browser to:  http://127.0.0.1:5000
echo Press Ctrl+C to stop.
echo.
%PYTHON% app.py
pause
