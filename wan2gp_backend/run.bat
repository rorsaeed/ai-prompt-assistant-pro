@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  Wan2GP Prompt Enhancer — Launch
::
::  Usage:
::    run.bat                          (default port 7860)
::    run.bat --port 7861
::    run.bat --share                  (public Gradio link)
::    run.bat --models-dir D:\models
::    run.bat --port 7861 --share
:: ============================================================

set "APP_DIR=%~dp0"
set "PYTHON_EXE=%APP_DIR%python_env\python.exe"
set "APP_SCRIPT=%APP_DIR%prompt_enhancer_app.py"

echo.
echo ============================================================
echo   Wan2GP Prompt Enhancer
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

if not exist "%APP_SCRIPT%" (
    echo [ERROR] Application script not found:
    echo         %APP_SCRIPT%
    echo.
    pause
    exit /b 1
)

:: ---- Launch the app ----------------------------------------
echo  Python : %PYTHON_EXE%
echo  Script : %APP_SCRIPT%
if not "%*"=="" echo  Args   : %*
echo.
echo  Starting ... (Ctrl+C to stop)
echo.

"%PYTHON_EXE%" "%APP_SCRIPT%" %*

:: Keep window open if the app exits with an error
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error ^(code %ERRORLEVEL%^).
    pause
)

endlocal
