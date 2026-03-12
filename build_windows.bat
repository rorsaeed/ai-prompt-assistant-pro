@echo off
echo Building AI Prompt Assistant for Windows...
flutter build windows --release

if %errorlevel% equ 0 (
    echo.
    echo Build successful!
    echo.
    echo Executable location: build\windows\runner\Release\ai_prompt_assistant.exe
    echo.
    pause
) else (
    echo.
    echo Build failed. Please check the error messages above.
    echo.
    pause
)
