# Windows PowerShell Setup Script for CryptoShift
# Save as: setup.ps1

# Bypass execution policy for this script
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Change to project directory
cd cryptoshift

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Green
pip install -r requirements.txt

# Install package in editable mode
Write-Host "Installing CryptoShift in editable mode..." -ForegroundColor Green
pip install -e .

# Initialize database
Write-Host "Initializing database..." -ForegroundColor Green
python -c "from src.data.database import init_db; init_db(); print('Database initialized!')"

Write-Host "`n✅ Setup complete! You can now run:
`n  Terminal 1: python scripts/download_historical.py
`n  Terminal 2: python scripts/run_backtest.py  
`n  Terminal 3: python scripts/start_ingestion.py
`n  Terminal 4: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
`n  Terminal 5: streamlit run dashboard/app.py
" -ForegroundColor Cyan
