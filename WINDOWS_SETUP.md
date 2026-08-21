# Windows PowerShell Setup Guide for CryptoShift

## 🚨 Issues You Encountered

### Issue 1: `&&` operator not valid
**Problem**: PowerShell doesn't use `&&` like bash/cmd
```powershell
# ❌ WRONG (bash syntax)
pip install -r requirements.txt && pip install -e .

# ✅ CORRECT (PowerShell syntax)
pip install -r requirements.txt; pip install -e .
```

### Issue 2: `uvicorn` and `streamlit` not found
**Problem**: Dependencies not installed yet because installation failed
**Solution**: Install requirements first

### Issue 3: Execution Policy blocking scripts
**Problem**: PowerShell blocks scripts by default
**Solution**: Run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

---

## ✅ Easiest Setup: Use Batch File (Recommended for Windows)

### Step 1: Open Command Prompt (NOT PowerShell)
- Press `Win+R`
- Type: `cmd.exe` and press Enter
- You're now in Command Prompt (not PowerShell)

### Step 2: Navigate to CryptoShift
```cmd
cd C:\NITK requirements\CPP\Test Project\cryptoshift
```

### Step 3: Run the setup batch file
```cmd
setup.bat
```

✅ **Done!** The batch file will:
- Create virtual environment
- Activate it
- Install all dependencies
- Initialize database

---

## 🔧 Manual Setup: PowerShell Method

If you prefer PowerShell, follow these exact steps:

### Step 1: Open PowerShell as Administrator
- Right-click PowerShell
- Select "Run as Administrator"

### Step 2: Allow script execution
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

### Step 3: Navigate to CryptoShift folder
```powershell
cd "C:\NITK requirements\CPP\Test Project\cryptoshift"
```

### Step 4: Run the PowerShell setup script
```powershell
.\setup.ps1
```

✅ **Done!** This will install everything automatically.

---

## 🚀 Manual Installation (Step-by-Step)

If scripts don't work, do this manually:

### Step 1: Create Virtual Environment
```powershell
python -m venv venv
```

### Step 2: Activate Virtual Environment

**In PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\venv\Scripts\Activate.ps1
```

**In Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**Expected result**: You should see `(venv)` prefix in your prompt
```
(venv) PS C:\NITK requirements\CPP\Test Project\cryptoshift>
```

### Step 3: Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### Step 4: Install requirements
```powershell
pip install -r requirements.txt
```

⏳ **Wait**: This takes 2-3 minutes (downloading 19 packages)

### Step 5: Install CryptoShift package
```powershell
pip install -e .
```

### Step 6: Initialize database
```powershell
python -c "from src.data.database import init_db; init_db()"
```

### Step 7: Verify installation
```powershell
pip list | Select-String -Pattern "fastapi|streamlit|uvicorn"
```

You should see:
```
fastapi                    0.141.1
streamlit                  1.62.0
uvicorn                    0.52.4
```

---

## 🎯 Running the System

After installation, open **5 separate terminal windows**:

### Terminal 1: Download Historical Data
```powershell
python scripts/download_historical.py
```
⏳ Wait for: "Historical data download completed!"

### Terminal 2: Run Backtesting
```powershell
python scripts/run_backtest.py
```
⏳ Wait for: "Backtesting completed!"

### Terminal 3: Start Real-Time Ingestion
```powershell
python scripts/start_ingestion.py
```
✅ Should show: "Fetching current data..." every 60 seconds

### Terminal 4: Start API Server
```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
✅ Should show: "Uvicorn running on http://0.0.0.0:8000"

### Terminal 5: Open Dashboard
```powershell
streamlit run dashboard/app.py
```
✅ Should show: "You can now view your Streamlit app in your browser"  
🌐 Open: http://localhost:8501

---

## 🧪 Test Installation

### Test 1: Check Python environment
```powershell
python --version
# Expected: Python 3.10+ (e.g., Python 3.11.9)
```

### Test 2: Check installed packages
```powershell
pip list | Select-String pandas
# Should show: pandas 3.0.5 (or similar)
```

### Test 3: Test API manually
```powershell
curl http://localhost:8000/api/health
# Expected response:
# {"status":"ok","timestamp":"...","uptime_seconds":...}
```

### Test 4: Run unit tests
```powershell
pytest tests/ -v
# Expected: 34+ tests passing
```

---

## 🆘 Troubleshooting

### Problem: "venv\Scripts\activate.bat is not recognized"
**Solution**: Make sure you're in the `cryptoshift` folder
```powershell
cd cryptoshift
dir venv/Scripts/  # Verify venv folder exists
```

### Problem: "ModuleNotFoundError: No module named 'src'"
**Solution**: Install package in editable mode:
```powershell
pip install -e .
```

### Problem: "Port 8000 already in use"
**Solution**: Use a different port
```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

### Problem: "No such file or directory: requirements.txt"
**Solution**: Make sure you're in the `cryptoshift` folder
```powershell
pwd  # Should show: ...\cryptoshift
ls requirements.txt  # Should exist
```

### Problem: "The term 'streamlit' is not recognized"
**Solution**: Make sure virtual environment is activated
```powershell
# You should see (venv) in your prompt
# If not, run:
.\venv\Scripts\Activate.ps1
```

---

## 📊 Expected Output After Setup

### After `pip install -r requirements.txt`:
```
Successfully installed pandas-3.0.5 numpy-2.4.6 scikit-learn-1.9.0 
scipy-1.17.1 fastapi-0.141.1 uvicorn-0.52.4 pydantic-2.13.4 
sqlalchemy-2.0.52 apscheduler-3.11.3 streamlit-1.62.0 plotly-6.9.0 
pycoingecko-3.2.0 requests-2.34.2 pytest-9.1.1 pytest-cov-7.1.0 
httpx-0.28.1 python-dotenv-1.2.3 pyyaml-6.0.3 tqdm-4.70.0
```

### After running API (Terminal 4):
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### After opening Dashboard (Terminal 5):
```
You can now view your Streamlit app in your browser.

Network URL: http://192.168.x.x:8501
Local URL: http://localhost:8501
```

---

## ✅ Quick Reference: PowerShell vs Command Prompt

| Task | PowerShell | Command Prompt |
|------|-----------|---|
| Activate venv | `.\venv\Scripts\Activate.ps1` | `venv\Scripts\activate.bat` |
| Chain commands | `;` separator | `&&` separator |
| List files | `ls` or `Get-ChildItem` | `dir` |
| Find in list | `Select-String` or `Where-Object` | `findstr` |
| Echo text | `Write-Host` | `echo` |

---

## 🎓 Key Differences: PowerShell vs Bash

Since you got bash syntax errors, here's a quick reference:

```powershell
# ❌ BASH (Linux/Mac)
pip install -r requirements.txt && pip install -e .
source venv/bin/activate

# ✅ POWERSHELL (Windows)
pip install -r requirements.txt; pip install -e .
.\venv\Scripts\Activate.ps1

# ✅ COMMAND PROMPT (Windows)
pip install -r requirements.txt && pip install -e .
venv\Scripts\activate.bat
```

**TIP**: If you're more comfortable with bash-style syntax, use **Command Prompt (cmd.exe)** instead of PowerShell!

---

## 🚀 Next Steps

1. ✅ Run `setup.bat` (easiest) OR `setup.ps1` (if using PowerShell)
2. ✅ Open 5 terminals and run the commands above
3. ✅ Visit http://localhost:8501 to see the dashboard
4. ✅ Check http://localhost:8000/docs for API documentation

**Questions?** Check the main [README.md](README.md) or [QUICK_START.md](QUICK_START.md)

Happy anomaly detecting! 🚀
