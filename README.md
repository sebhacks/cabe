# 🚀 CABE (Cool Ass Backtest Engine)

**CABE** is a stock trading analysis tool that provides VWAP and RSI technical analysis using real-time market data from Polygon.io.

## 🛠️ Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get your Polygon.io API key**
   - Visit [Polygon.io](https://polygon.io/) and sign up
   - Copy your API key from the dashboard

3. **Create environment file**
   ```bash
   echo "POLYGON_API_KEY=your_api_key_here" > .env
   ```

## 🚀 Usage

Analyze a stock for a specific date range:

```bash
python3 cabe.py SPY 062525 062727
```

**Parameters:**
- `SPY` - Stock symbol (uppercase, 1-5 characters)
- `062525` - Start date (MMDDYY format: June 25, 2025)
- `062727` - End date (MMDDYY format: June 27, 2025)

## 📈 Output

CABE provides:
- Daily OHLCV data
- VWAP calculation and cross detection
- RSI calculation with overbought/oversold analysis
- Summary statistics for the date range

## 📁 Files

```
cabe/
├── cabe.py           # Main analysis tool
├── requirements.txt  # Python dependencies
├── README.md        # This file
└── .env             # API key (create this)
```

## 🧪 Examples

```bash
# Analyze SPY
python3 cabe.py SPY 062525 062727

# Analyze Apple
python3 cabe.py AAPL 062525 062725

# Analyze Tesla
python3 cabe.py TSLA 062025 062725
```

## ⚠️ Notes

- Use **MMDDYY** date format (e.g., `062525` for June 25, 2025)
- Stock symbols must be uppercase (e.g., SPY, AAPL, TSLA)
- Free Polygon.io tier has rate limits (5 calls/minute)
- Intraday data only available for recent trading days 