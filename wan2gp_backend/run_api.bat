@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  Wan2GP Prompt Enhancer — API Server Launch  (for Flutter)
::
::  Usage:
::    run_api.bat                          (default port 7860)
::    run_api.bat --port 7861
::    run_api.bat --models-dir D:\models
:: ============================================================

set "APP_DIR=%~dp0"
set "PYTHON_EXE=%APP_DIR%python_env\python.exe"
set "APP_SCRIPT=%APP_DIR%api_server.py"

echo.
echo ============================================================
echo   Wan2GP Prompt Enhancer  —  API Server
echo ============================================================
echo.

:: ---- Check that setup was run ------------------------------
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python environment not found.
    echo         Please run  setup.bat  first.
    echo.
    pause
    exit /b 1
)

:: ---- Launch ------------------------------------------------
echo Starting API server …
echo.
"%PYTHON_EXE%" "%APP_SCRIPT%" %*
echo.
pause
