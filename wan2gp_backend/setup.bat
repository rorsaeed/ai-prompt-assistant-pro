@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  Wan2GP Prompt Enhancer — One-time Setup
::
::  Downloads Python 3.11 embeddable into python_env\,
::  bootstraps pip, and installs all required packages.
::  Re-run at any time to upgrade packages.
:: ============================================================

set "APP_DIR=%~dp0"
set "PYTHON_DIR=%APP_DIR%python_env"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "PIP_EXE=%PYTHON_DIR%\Scripts\pip.exe"

echo.
echo ============================================================
echo   Wan2GP Prompt Enhancer  -  Setup
echo ============================================================
echo.

:: ---- Check if Python env already exists --------------------
if exist "%PYTHON_EXE%" (
    echo [OK] Python environment already exists at:
    echo      %PYTHON_DIR%
    echo      Skipping download — running package install only.
    echo.
    goto :install_packages
)

:: ---- 1. Download Python 3.11 embeddable (Windows 64-bit) ---
echo [1/5] Downloading Python 3.11.9 embeddable package ...
set "PY_ZIP=%TEMP%\python-3.11-embed-amd64.zip"
set "PY_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"

powershell -NoProfile -Command ^
  "Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_ZIP%' -UseBasicParsing"
if errorlevel 1 (
    echo.
    echo [ERROR] Could not download Python. Check your internet connection.
    pause & exit /b 1
)
echo [OK] Downloaded.

:: ---- 2. Extract Python -------------------------------------
echo.
echo [2/5] Extracting Python to:
echo      %PYTHON_DIR%
if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"
powershell -NoProfile -Command ^
  "Expand-Archive -Path '%PY_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
del /q "%PY_ZIP%"

:: Enable site-packages (required for installed packages to be importable)
set "PTH_FILE=%PYTHON_DIR%\python311._pth"
if exist "%PTH_FILE%" (
    powershell -NoProfile -Command ^
      "(Get-Content '%PTH_FILE%') -replace '^#import site','import site' | Set-Content '%PTH_FILE%'"
    echo [OK] Enabled site-packages.
)

:: Download Python dev headers/libs (needed by Triton for kernel compilation)
echo  ^> Fetching Python dev headers for Triton ...
set "NUGET_ZIP=%TEMP%\python3119_nuget.zip"
set "NUGET_DIR=%TEMP%\python3119_nuget"
powershell -NoProfile -Command ^
  "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.nuget.org/api/v2/package/python/3.11.9' -OutFile '%NUGET_ZIP%' -UseBasicParsing"
if errorlevel 1 (
    echo [WARN] Could not download Python dev headers — Triton may not work.
) else (
    if exist "%NUGET_DIR%" rd /s /q "%NUGET_DIR%"
    powershell -NoProfile -Command "Expand-Archive -Path '%NUGET_ZIP%' -DestinationPath '%NUGET_DIR%' -Force"
    xcopy "%NUGET_DIR%\tools\include" "%PYTHON_DIR%\Include\" /E /I /Y >nul 2>&1
    xcopy "%NUGET_DIR%\tools\libs"    "%PYTHON_DIR%\libs\"    /E /I /Y >nul 2>&1
    del /q "%NUGET_ZIP%"
    rd /s /q "%NUGET_DIR%"
    echo [OK] Python dev headers installed.
)

:: ---- 3. Bootstrap pip --------------------------------------
echo.
echo [3/5] Bootstrapping pip ...
set "GET_PIP=%TEMP%\get-pip.py"
powershell -NoProfile -Command ^
  "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GET_PIP%' -UseBasicParsing"
if errorlevel 1 (
    echo [ERROR] Could not download get-pip.py.
    pause & exit /b 1
)
"%PYTHON_EXE%" "%GET_PIP%"
if errorlevel 1 ( echo [ERROR] pip bootstrap failed. & pause & exit /b 1 )
del /q "%GET_PIP%"
echo [OK] pip ready.

:install_packages
:: ---- 4. Install packages -----------------------------------
echo.
echo [4/5] Installing packages (first run may take 10-15 minutes) ...
echo.

:: --- PyTorch 2.10 with CUDA 13.0 ---
echo  ^> PyTorch ^(CUDA 13.0^) ...
"%PIP_EXE%" install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 (
    echo  [WARN] CUDA wheel failed — falling back to CPU-only torch.
    "%PIP_EXE%" install torch
)

:: --- All requirements from requirements.txt ---
echo  ^> Installing requirements.txt ...
"%PIP_EXE%" install -r "%APP_DIR%requirements.txt"
if errorlevel 1 ( echo [ERROR] requirements install failed. & pause & exit /b 1 )

:: --- Acceleration packages (optional but recommended) ---
echo.
echo [5/5] Installing acceleration packages ...
echo.

echo  ^> Triton ...
"%PIP_EXE%" install -U triton-windows
if errorlevel 1 ( echo  [WARN] Triton install failed — Sage/Flash attention will be unavailable. )

echo  ^> Flash Attention 2 ...
"%PIP_EXE%" install https://github.com/deepbeepmeep/kernels/releases/download/Flash2/flash_attn-2.8.3-cp311-cp311-win_amd64.whl
if errorlevel 1 ( echo  [WARN] Flash Attention install failed — will use SDPA fallback. )

echo  ^> SageAttention 2 ...
"%PIP_EXE%" install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
if errorlevel 1 ( echo  [WARN] SageAttention install failed — will use SDPA fallback. )

echo  ^> GGUF llama.cpp CUDA kernels ...
"%PIP_EXE%" install https://github.com/deepbeepmeep/kernels/releases/download/GGUF_Kernels/llamacpp_gguf_cuda-1.0.2+torch210cu13py311-cp311-cp311-win_amd64.whl
if errorlevel 1 ( echo  [WARN] GGUF CUDA kernels install failed — GGUF will use slower CPU fallback. )

echo.
echo ============================================================
echo   Setup complete!
echo   Run  run.bat  to launch the Prompt Enhancer.
echo ============================================================
echo.
pause
endlocal
