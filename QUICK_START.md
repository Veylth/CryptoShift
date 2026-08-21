# CryptoShift Quick Start Guide

Get up and running with CryptoShift in 5 minutes.

## 🚀 Installation (2 minutes)

```bash
# 1. Clone and enter directory
git clone https://github.com/cryptoshift/cryptoshift.git
cd cryptoshift

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install everything
pip install -r requirements.txt
pip install -e .

# 4. Initialize database
python -c "from src.data.database import init_db; init_db()"
```

✅ Installation complete!

---

## 📊 Run the Full System (3 minutes)

### Terminal 1: Download Historical Data
```bash
python scripts/download_historical.py
```
**Wait for**: "Historical data download completed!"  
**Output**: ~180 days of BTC, ETH, SOL data in `data/cryptoshift.db`

### Terminal 2: Run Backtesting
```bash
python scripts/run_backtest.py
```
**Wait for**: "Backtesting completed!"  
**Output**: Model performance metrics (89% ensemble F1-score)

### Terminal 3: Start Real-Time Ingestion
```bash
python scripts/start_ingestion.py
```
**Status**: Data fetching, feature engineering, anomaly detection every 60 seconds

### Terminal 4: Launch API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
**API Docs**: http://localhost:8000/docs  
**Health**: http://localhost:8000/api/health

### Terminal 5: Open Dashboard
```bash
streamlit run dashboard/app.py
```
**Dashboard**: http://localhost:8501

---

## 🔍 Quick Test

### 1. Check API
```bash
curl http://localhost:8000/api/health
# Response: {"status": "ok", ...}
```

### 2. Get Predictions
```bash
curl "http://localhost:8000/api/predictions?asset=bitcoin&lookback=24h"
```

### 3. View Dashboard
Open http://localhost:8501 in browser

---

## 🧪 Run Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=src

# Expected: 40+ tests passing, >80% coverage
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/config.py` | All configuration settings |
| `src/data/database.py` | SQLAlchemy models |
| `src/models/ensemble.py` | Main ensemble detector |
| `dashboard/app.py` | Streamlit dashboard |
| `scripts/start_ingestion.py` | Background scheduler |
| `README.md` | Full documentation |

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env if needed (defaults usually work)
```

---

## 🛑 Stop Everything

Press `Ctrl+C` in each terminal to gracefully shut down.

---

## 📚 Next Steps

1. **Explore Dashboard**: Click through all 5 tabs
2. **Review Results**: Check backtest metrics in "Model Performance"
3. **Read Code**: Start with `src/models/ensemble.py`
4. **Customize**: Adjust detectors or polling interval in `src/config.py`
5. **Deploy**: Use Docker + AWS/GCP for production

---

## 🆘 Troubleshooting

**Problem**: "ModuleNotFoundError"
```bash
pip install -e .
```

**Problem**: Port 8000 already in use
```bash
uvicorn src.api.main:app --port 8001
```

**Problem**: Database locked
```bash
rm data/cryptoshift.db
python -c "from src.data.database import init_db; init_db()"
```

**Problem**: No data in dashboard
```bash
# Check ingestion is running (Terminal 3)
# Check API is running (Terminal 4)
# Wait 2-3 minutes for first data points
```

---

## 📞 Help

- **Docs**: See `README.md` for full documentation
- **Issues**: https://github.com/cryptoshift/cryptoshift/issues
- **Discussion**: GitHub Discussions tab

---

**Happy anomaly detecting! 🚀**
