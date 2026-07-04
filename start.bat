@echo off
REM ── Truth Engine Quick Start (Windows) ────────────────────────────────────
echo.
echo ╔══════════════════════════════════════════════╗
echo ║          Truth Engine — Quick Start          ║
echo ╚══════════════════════════════════════════════╝
echo.

REM Check .env
if not exist .env (
    echo Copying .env.example to .env ...
    copy .env.example .env
    echo.
    echo ⚠  Please edit .env and add your GEMINI_API_KEY and GROQ_API_KEY
    echo    Then run this script again.
    pause
    exit /b 1
)

REM Check venv
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate and install
call venv\Scripts\activate.bat
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo ✓ Dependencies ready.
echo.

REM Start backend in a new window
echo Starting FastAPI backend on http://localhost:8000 ...
start "Truth Engine - Backend" cmd /k "venv\Scripts\activate && python main.py"

REM Wait for backend to warm up
timeout /t 5 /nobreak > nul

REM Start Streamlit
echo Starting Streamlit UI on http://localhost:8501 ...
start "Truth Engine - Frontend" cmd /k "venv\Scripts\activate && streamlit run streamlit_app.py"

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Streamlit UI  →  http://localhost:8501
echo   API Docs      →  http://localhost:8000/docs
echo   Health check  →  http://localhost:8000/health
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
pause
