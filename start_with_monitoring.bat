@echo off
REM Script to start the CAMEL Extensions application with monitoring enabled on Windows

REM Set environment variables for monitoring
set LOG_LEVEL=INFO
set PROMETHEUS_PORT=9090
set ENABLE_PROMETHEUS=true

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please install Python 3.8 or higher.
    exit /b 1
)

REM Check if required directories exist
if not exist backend (
    echo Error: backend directory not found. Please run this script from the project root.
    exit /b 1
)
if not exist gui (
    echo Error: gui directory not found. Please run this script from the project root.
    exit /b 1
)

REM Install all dependencies if needed
if not exist .deps_installed (
    echo Installing project dependencies...
    pip install -r requirements.txt
    echo. > .deps_installed
)

REM Start the backend server
echo Starting backend server with monitoring enabled...
start "CAMEL Extensions Backend" cmd /c "cd backend && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for backend to start
echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

REM Start the frontend
echo Starting frontend...
start "CAMEL Extensions Frontend" cmd /c "cd gui && streamlit run app.py"

echo Services started:
echo - Backend: http://localhost:8000
echo - Frontend: http://localhost:8501
echo - Prometheus metrics: http://localhost:9090
echo - API metrics: http://localhost:8000/api/metrics
echo.
echo Close the command windows to stop the services

REM Keep the script running
echo Press Ctrl+C to exit this window (services will continue running)
pause > nul