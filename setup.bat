@echo off
title NMR ^& IR Predictor — First-Time Setup
echo ============================================
echo   NMR ^& IR Predictor — Setup
echo ============================================
echo.
echo This will install all required Python packages.
echo Requires Python 3.10+ with pip.
echo.

REM Detect Python
where python >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=python
    goto :install
)
where py >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=py
    goto :install
)

echo ERROR: Python not found.
echo Download from https://python.org  (check "Add to PATH")
pause
exit /b 1

:install
echo Using: %PYTHON%
echo.

REM Upgrade pip silently
%PYTHON% -m pip install --upgrade pip --quiet

echo Installing dependencies (this may take a few minutes)...
echo  - Flask
echo  - NumPy
echo  - RDKit
echo.

%PYTHON% -m pip install -r requirements.txt

if %errorlevel%==0 (
    echo.
    echo ============================================
    echo   Setup complete!
    echo   Run  run.bat  to start the app.
    echo ============================================
) else (
    echo.
    echo Setup encountered errors. Check the output above.
)
echo.
pause
