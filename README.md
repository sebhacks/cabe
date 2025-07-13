# 🚀 CABE (Cool Ass Backtest Engine)

**CABE** is a stock analysis tool that provides backtesting capabilities for Daily VWAP, Bollinger Band, and RSI crossovers during a given time period. The tool uses historical data fetched from Polygon.io, for which you will need to set up a (free or paid) account and get your own API token.

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

### Basic Usage

Analyze a stock for a specific date range:

```bash
python3 cabe.py SPY 062525 062727
```

### Command-Line Arguments

**Required Arguments:**
- `symbol` - Stock symbol (uppercase, 1-5 characters, e.g., SPY, AAPL, TSLA)
- `start_date` - Start date (MMDDYY format: June 25, 2025 = 062525)
- `end_date` - End date (MMDDYY format: June 27, 2025 = 062727)

**Optional Flags:**
- `--window <days>` - Event window size in days for return calculations (default: 1)
- `--no-viz` - Disable visualization generation (saves time and disk space)
- `--capital <amount>` - Initial capital for trade simulation (default: 100000)
- `--futures` - Enable futures symbol support (required for /MES, /MNQ, etc.)

### Examples

```bash
# Basic analysis
python3 cabe.py SPY 062525 062727

# With custom window size (3 days)
python3 cabe.py AAPL 062525 062725 --window 3

# Disable visualizations (faster execution)
python3 cabe.py TSLA 062025 062725 --no-viz

# Custom initial capital for simulation
python3 cabe.py SPY 062525 062727 --capital 50000

# Combine multiple options
python3 cabe.py AAPL 062525 062725 --window 5 --no-viz --capital 250000

# Futures symbols (requires Polygon futures data access)
python3 cabe.py /MES 062525 062727 --futures
python3 cabe.py /MNQ 062525 062727 --futures
```

## 📈 Analysis Features

CABE provides comprehensive technical analysis including:

### Technical Indicators
- **VWAP (Volume Weighted Average Price)** - Calculated using daily OHLCV data
- **RSI (Relative Strength Index)** - 14-period RSI with overbought/oversold levels (80/20)
- **Bollinger Bands** - 20-period bands with 2 standard deviations
- **Volume Analysis** - Volume confirmation for signal validation

### Event Detection
- **VWAP Crosses** - Price crossing above/below VWAP with directional tracking
- **RSI Crosses** - RSI crossing overbought (80) or oversold (20) levels
- **Bollinger Band Crosses** - Price breaking above upper or below lower bands with volume confirmation

### Trade Simulation
- Simulates buying on each detected event
- Calculates returns after specified window period
- Tracks cumulative P&L and trade statistics
- Provides directional breakdown of performance

## 📁 Output Files

CABE generates several output files in the `cabe_output/` directory:

### Data Files
- `{SYMBOL}_{START_DATE}_{END_DATE}_events.csv` - All detected events with return calculations and directional information
- `{SYMBOL}_{START_DATE}_{END_DATE}_trades.csv` - Individual trade simulation results

### Visualization Files (unless `--no-viz` is used)
- `{SYMBOL}_{DATE}_chart.png` - Daily charts showing:
  - Price action with VWAP and Bollinger Bands
  - RSI indicator with overbought/oversold levels
  - Volume bars

### File Naming Convention
- Events/Trades: `SPY_2025-06-25_2025-06-27_events.csv`
- Charts: `SPY_20250625_chart.png` (date format: YYYYMMDD)

## 📊 Output Summary

CABE provides a comprehensive summary with directional breakdown including:

### Event Statistics (Directional)
- **VWAP Crosses**: Above VWAP vs Below VWAP performance
- **RSI Signals**: Overbought (>80) vs Oversold (<20) performance  
- **Bollinger Band Breakouts**: Above Upper Band vs Below Lower Band performance
- Win rate percentage for each direction
- Average and median returns by direction

### Trade Simulation Results
- Total number of trades executed
- Total P&L (profit/loss)
- Final capital after all trades
- Average and median trade returns

### Example Output
```
================================================================================
📊 ANALYSIS RESULTS FOR SPY
📅 Date Range: 2025-06-25 to 2025-06-27
📈 Trading Days: 3
================================================================================

🎯 EVENT DETECTION SUMMARY
--------------------------------------------------------------------------------

💹 VWAP CROSSES:
  Crossed Above VWAP        | Count: 8  | Win Rate: 75.0% | Avg Return: 0.045% | Median: 0.038%
  Crossed Below VWAP        | Count: 3  | Win Rate: 33.3% | Avg Return: -0.012% | Median: -0.008%

📉 RSI SIGNALS:
  RSI Overbought (>80)      | Count: 2  | Win Rate: 50.0% | Avg Return: -0.008% | Median: -0.006%
  RSI Oversold (<20)        | Count: 5  | Win Rate: 80.0% | Avg Return: 0.052% | Median: 0.045%

📊 BOLLINGER BAND BREAKOUTS:
  Broke Above Upper Band    | Count: 1  | Win Rate: 100.0% | Avg Return: 0.078% | Median: 0.078%
  Broke Below Lower Band    | Count: 0  | Count: 0

================================================================================
💰 TRADE SIMULATION RESULTS
================================================================================
  📈 Total Trades: 19
  💵 Initial Capital: $100,000.00
  📊 Total P&L: $1,234.56
  🎯 Final Capital: $101,234.56
  📈 Avg Trade Return: 0.065%
  📊 Median Trade Return: 0.052%
================================================================================
```

## ⚠️ Important Notes

### Date Format
- Use **MMDDYY** date format (e.g., `062525` for June 25, 2025)
- Ensure dates are valid trading days
- End date should be after start date

### Stock Symbols
- Must be uppercase (e.g., SPY, AAPL, TSLA)
- Limited to 1-5 characters for stocks
- Futures symbols supported (e.g., /MES, /MNQ) - requires Polygon futures data access
- Must be valid symbols available on Polygon.io

### API Limitations
- Free Polygon.io tier has rate limits (5 calls/minute)
- CABE includes automatic retry logic with 5-second delays for rate limit handling
- Uses efficient single daily data calls to minimize API usage
- Futures data requires appropriate Polygon.io subscription level

### Performance Considerations
- Use `--no-viz` flag for faster execution when visualizations aren't needed
- Larger date ranges require more API calls and processing time
- Window size affects return calculations and simulation results
- Optimized for minimal API calls with daily data aggregation

### Data Requirements
- Minimum 20 data points required for Bollinger Bands calculation
- Sufficient volume data needed for signal confirmation
- Events only detected when enough historical data is available
- Cross detection uses previous day comparison for accurate signals

## 🧪 Advanced Usage Examples

```bash
# Long-term analysis with 5-day window
python3 cabe.py SPY 010125 123125 --window 5 --capital 1000000

# Quick analysis without visualizations
python3 cabe.py AAPL 062525 062725 --no-viz

# High-frequency analysis with 1-day window
python3 cabe.py TSLA 062525 062725 --window 1 --capital 50000

# Multiple symbols analysis (run separately)
python3 cabe.py SPY 062525 062727
python3 cabe.py QQQ 062525 062727
python3 cabe.py IWM 062525 062727

# Futures analysis (requires appropriate Polygon subscription)
python3 cabe.py /MES 062525 062727 --futures --window 3
python3 cabe.py /MNQ 062525 062727 --futures --no-viz
```

## 📁 Project Structure

```
cabe/
├── cabe.py           # Main analysis tool
├── analytics.py      # Technical analysis functions
├── requirements.txt  # Python dependencies
├── README.md        # This file
├── .env             # API key (create this)
└── cabe_output/     # Generated output files
    ├── *_events.csv # Event detection results
    ├── *_trades.csv # Trade simulation results
    └── *_chart.png  # Visualization charts
```

## 🔧 Recent Updates

- **Directional Analysis**: Added detailed breakdown of VWAP and Bollinger Band crosses by direction
- **Futures Support**: Added support for futures symbols (e.g., /MES, /MNQ)
- **API Optimization**: Reduced API calls by using single daily data aggregation
- **Enhanced Error Handling**: Improved input validation and rate limit handling
- **Memory Optimization**: Efficient data processing for large date ranges
- **Security Improvements**: Better API key handling and error logging
