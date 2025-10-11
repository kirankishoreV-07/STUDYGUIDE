@echo off
title StudyHub - Unified AI Learning Platform
color 0A

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                    StudyHub Launcher                         ║
echo  ║              Unified AI Learning Platform                    ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo [INFO] Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo [INFO] Python detected. Checking StudyHub installation...
echo.

:: Check if we're in the right directory
if not exist "app.py" (
    echo [ERROR] app.py not found! Please run this script from the StudyHub directory.
    pause
    exit /b 1
)

:: Check if .env file exists
if not exist ".env" (
    echo [WARNING] .env file not found. Creating from template...
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [INFO] Created .env file. Please edit it with your API keys.
    ) else (
        echo [ERROR] .env.example not found!
    )
)

:: Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if requirements are installed
echo [INFO] Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies!
        pause
        exit /b 1
    )
) else (
    echo [INFO] Dependencies already installed.
)

echo.
echo [SUCCESS] Setup complete! Starting StudyHub...
echo.

:: Check if Ollama is running
echo [INFO] Checking Ollama service...
curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama not detected. PDF Chat may not work fully.
    echo [INFO] To install Ollama: https://ollama.ai
    echo [INFO] Then run: ollama pull mistral
) else (
    echo [INFO] Ollama detected and running.
)

echo.
echo ┌─────────────────────────────────────────────────────────────┐
echo │                    StudyHub is starting...                  │
echo │                                                             │
echo │  🌐 Web Interface: http://localhost:5000                   │
echo │  📚 PDF Chat: http://localhost:5000/pdf-chat               │
echo │  🎬 Summarizer: http://localhost:5000/summarizer           │
echo │  ❓ Quiz Generator: http://localhost:5000/quiz-generator    │
echo │                                                             │
echo │  Press Ctrl+C to stop the server                          │
echo └─────────────────────────────────────────────────────────────┘
echo.

:: Start the Flask application
python app.py

:: Cleanup
echo.
echo [INFO] StudyHub stopped.
pause