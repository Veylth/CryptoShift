@echo off
REM Windows Batch Setup Script for CryptoShift
REM Run this file from Command Prompt (cmd.exe), not PowerShell

echo.
echo Setting up CryptoShift...
echo.

cd cryptoshift

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements (this may take 2-3 minutes)...
pip install -r requirements.txt

REM Install package in editable mode
echo Installing CryptoShift package...
pip install -e .

REM Initialize database
echo Initializing database...
python -c "from src.data.database import init_db; init_db(); print('Database initialized!')"

echo.
echo ======================================
echo   Setup complete!
echo ======================================
echo.
echo Run these in separate terminals:
echo   1. python scripts/download_historical.py
echo   2. python scripts/run_backtest.py
echo   3. python scripts/start_ingestion.py
echo   4. uvicorn src.api.main:app --host 0.0.0.0 --port 8000
echo   5. streamlit run dashboard/app.py
echo.
pause
