#!/usr/bin/env python3
"""
CABE (Cool Ass Backtest Engine) - Professional Quantitative Trading Analysis Tool
Features: VWAP, RSI, Cross Detection, Data Visualization with Polygon.io API integration
"""

import os
import re
import json
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta, timezone
import argparse
import pandas as pd
import numpy as np
from polygon import RESTClient
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from analytics import detect_rsi_crosses, detect_vwap_crosses, annotate_event_window_return, aggregate_event_stats, simulate_trades

# Load API key securely
load_dotenv()
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

if not POLYGON_API_KEY:
    raise RuntimeError('Polygon.io API key not found. Please set POLYGON_API_KEY in your .env file.')

# Security: Validate API key format
if not re.match(r'^[A-Za-z0-9_-]+$', POLYGON_API_KEY):
    raise ValueError('Invalid API key format detected')

# Initialize Polygon client
client = RESTClient(api_key=POLYGON_API_KEY)

# Configure matplotlib for professional output
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TradingData:
    """Structured container for trading data analysis"""
    
    def __init__(self, symbol: str, date: str):
        self.symbol = symbol
        self.date = date
        self.ohlcv_data = None
        self.vwap = None
        self.rsi_values = None
        self.crosses = {
            'vwap': [],
            'rsi_overbought': [],
            'rsi_oversold': []
        }
        self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'date': self.date,
            'ohlcv_data': self.ohlcv_data,
            'vwap': self.vwap,
            'rsi_values': self.rsi_values,
            'crosses': self.crosses,
            'metrics': self.metrics
        }

def validate_symbol(symbol: str) -> bool:
    """Enhanced symbol validation for security"""
    if not symbol or len(symbol) > 10:
        return False
    return bool(re.match(r'^[A-Z]{1,5}$', symbol.upper()))

def parse_date(date_str: str) -> Optional[str]:
    """Parse date string to YYYY-MM-DD format"""
    try:
        if len(date_str) != 6:
            print("ERROR: Date must be in MMDDYY format (e.g., 061323 for June 13, 2023)")
            return None
        
        month = int(date_str[:2])
        day = int(date_str[2:4])
        year = int(date_str[4:6])
        
        # Security: Validate date ranges
        if not (1 <= month <= 12 and 1 <= day <= 31):
            print("ERROR: Invalid date values")
            return None
        
        if year < 50:
            year += 2000
        else:
            year += 1900
        
        # Validate the complete date
        datetime(year, month, day)
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        print("ERROR: Invalid date format. Please use MMDDYY format (e.g., 061323)")
        return None

def calculate_rsi_vectorized(prices: List[float], period: int = 14) -> List[Optional[float]]:
    """Vectorized RSI calculation for optimal performance"""
    if len(prices) < period + 1:
        return []
    
    # Convert to pandas Series for vectorized operations
    price_series = pd.Series(prices)
    
    # Calculate price changes
    deltas = price_series.diff()
    
    # Separate gains and losses
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    
    # Calculate exponential moving averages
    avg_gains = gains.ewm(span=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    # Convert to list and handle NaN values
    rsi_list = rsi.tolist()
    rsi_padded = [None] * period + [val if not pd.isna(val) else None for val in rsi_list[period:]]
    
    return rsi_padded

def detect_crosses_vectorized(values: List[float], threshold: float, timestamps: List[str]) -> List[Dict[str, Any]]:
    """Vectorized cross detection for optimal performance"""
    if len(values) < 2:
        return []
    
    # Convert to numpy arrays for vectorized operations
    values_array = np.array(values)
    valid_mask = ~pd.isna(values_array)
    
    if not np.any(valid_mask):
        return []
    
    # Find valid values
    valid_values = values_array[valid_mask]
    valid_timestamps = [timestamps[i] for i, valid in enumerate(valid_mask) if valid]
    
    if len(valid_values) < 2:
        return []
    
    # Vectorized cross detection
    above_threshold = valid_values > threshold
    crosses = []
    
    for i in range(1, len(valid_values)):
        if above_threshold[i] != above_threshold[i-1]:
            crosses.append({
                'timestamp': valid_timestamps[i],
                'value': float(valid_values[i]),
                'direction': 'above' if above_threshold[i] else 'below',
                'threshold': threshold
            })
    
    return crosses

def calculate_vwap_vectorized(highs: List[float], lows: List[float], 
                            closes: List[float], volumes: List[float]) -> float:
    """Vectorized VWAP calculation for optimal performance"""
    # Convert to numpy arrays for vectorized operations
    highs_array = np.array(highs)
    lows_array = np.array(lows)
    closes_array = np.array(closes)
    volumes_array = np.array(volumes)
    
    # Calculate typical prices vectorized
    typical_prices = (highs_array + lows_array + closes_array) / 3
    
    # Calculate VWAP using weighted average
    return float(np.average(typical_prices, weights=volumes_array))

def count_vwap_crosses_vectorized(closes: List[float], vwap: float) -> int:
    """Vectorized VWAP cross counting for optimal performance"""
    if len(closes) < 2:
        return 0
    
    # Convert to numpy array for vectorized operations
    closes_array = np.array(closes)
    
    # Vectorized cross detection
    above_vwap = closes_array > vwap
    crosses = np.sum(np.diff(above_vwap.astype(int)))
    
    return int(abs(crosses))

def process_aggregation_data(aggs: List) -> Optional[Dict[str, List]]:
    """Efficiently process aggregation data using vectorized operations"""
    if not aggs:
        return None
    
    # Extract all data in one pass using vectorized operations
    data = np.array([[agg.open, agg.high, agg.low, agg.close, agg.volume, agg.timestamp] for agg in aggs])
    
    opens = data[:, 0].tolist()
    highs = data[:, 1].tolist()
    lows = data[:, 2].tolist()
    closes = data[:, 3].tolist()
    volumes = data[:, 4].tolist()
    timestamps = [datetime.fromtimestamp(ts/1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S') for ts in data[:, 5]]
    
    return {
        'opens': opens,
        'highs': highs,
        'lows': lows,
        'closes': closes,
        'volumes': volumes,
        'timestamps': timestamps
    }

def fetch_with_rate_limit_handling(fetch_fn, *args, **kwargs) -> List:
    """Handle Polygon API rate limits with automatic retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return list(fetch_fn(*args, **kwargs))
        except Exception as e:
            # Check for rate limit errors
            error_str = str(e).lower()
            if 'rate limit' in error_str or '429' in error_str or 'too many requests' in error_str:
                print(f"RATE_LIMIT: Polygon API rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting 65 seconds...")
                time.sleep(65)
            else:
                raise
    print("ERROR: Failed after multiple retries due to rate limits or other errors.")
    return []

def analyze_intraday_data(symbol: str, date_str: str) -> Optional[TradingData]:
    """Analyze intraday data for a specific date"""
    try:
        # Security: Validate symbol
        if not validate_symbol(symbol):
            print(f"ERROR: Invalid symbol format: {symbol}")
            return None
        
        # Get intraday data (5-minute bars) with rate limit handling
        aggs = fetch_with_rate_limit_handling(
            client.list_aggs,
            ticker=symbol,
            multiplier=5,
            timespan="minute",
            from_=date_str,
            to=date_str,
            limit=50000
        )
        
        if not aggs:
            print(f"NO_DATA: No intraday data available for {date_str}")
            return None
        
        # Process data efficiently
        data = process_aggregation_data(aggs)
        if not data:
            return None
        
        # Create trading data object
        trading_data = TradingData(symbol, date_str)
        trading_data.ohlcv_data = data
        
        # Calculate VWAP using vectorized operations
        trading_data.vwap = calculate_vwap_vectorized(
            data['highs'], data['lows'], data['closes'], data['volumes']
        )
        
        # Calculate RSI using vectorized operations
        trading_data.rsi_values = calculate_rsi_vectorized(data['closes'])
        
        # Detect crosses using vectorized operations
        vwap_cross_count = count_vwap_crosses_vectorized(data['closes'], trading_data.vwap)
        trading_data.crosses['vwap'] = [{'timestamp': data['timestamps'][0], 'value': trading_data.vwap, 'direction': 'cross', 'threshold': trading_data.vwap} for _ in range(vwap_cross_count)]
        trading_data.crosses['rsi_overbought'] = detect_rsi_crosses(trading_data.rsi_values, 80, data['timestamps'])
        trading_data.crosses['rsi_oversold'] = detect_rsi_crosses(trading_data.rsi_values, 20, data['timestamps'])
    
        # Calculate additional metrics
        trading_data.metrics = {
            'price_volatility': float(np.std(data['closes'])),
            'volume_avg': float(np.mean(data['volumes'])),
            'price_range': float(max(data['highs']) - min(data['lows'])),
            'total_volume': float(sum(data['volumes']))
        }
        
        return trading_data
        
    except Exception as e:
        print(f"ERROR: Error fetching intraday data: {e}")
        return None

def create_visualizations(trading_data: TradingData, output_dir: str = "cabe_output"):
    """Create professional trading visualizations"""
    if not trading_data.ohlcv_data:
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    df = pd.DataFrame({
        'timestamp': trading_data.ohlcv_data['timestamps'],
        'open': trading_data.ohlcv_data['opens'],
        'high': trading_data.ohlcv_data['highs'],
        'low': trading_data.ohlcv_data['lows'],
        'close': trading_data.ohlcv_data['closes'],
        'volume': trading_data.ohlcv_data['volumes']
    })
    
    if trading_data.rsi_values:
        df['rsi'] = trading_data.rsi_values
    
    # Convert timestamps to datetime for plotting
    df['datetime'] = pd.to_datetime(df['timestamp'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'{trading_data.symbol} - {trading_data.date} - Trading Analysis', fontsize=16, fontweight='bold')
    
    # Price and VWAP chart
    axes[0].plot(df['datetime'], df['close'], label='Close Price', linewidth=1.5, color='#1f77b4')
    if trading_data.vwap:
        axes[0].axhline(y=trading_data.vwap, color='red', linestyle='--', alpha=0.7, label=f'VWAP: ${trading_data.vwap:.2f}')
    axes[0].set_ylabel('Price ($)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RSI chart
    if trading_data.rsi_values:
        axes[1].plot(df['datetime'], df['rsi'], label='RSI', linewidth=1.5, color='#ff7f0e')
        axes[1].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Overbought (80)')
        axes[1].axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Oversold (20)')
        axes[1].set_ylabel('RSI', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Volume chart
    axes[2].bar(df['datetime'], df['volume'], alpha=0.7, color='#2ca02c', label='Volume')
    axes[2].set_ylabel('Volume', fontweight='bold')
    axes[2].set_xlabel('Time', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    filename = f"{output_dir}/{trading_data.symbol}_{trading_data.date}_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"VISUALIZATION: Saved to {filename}")

def analyze_date_range(symbol: str, start_date: str, end_date: str, 
                      save_data: bool = True, create_viz: bool = True, window: int = 1) -> Dict[str, Any]:
    """Analyze stock data for a date range with structured output"""
    # Security: Validate inputs
    if not validate_symbol(symbol):
        print(f"ERROR: Invalid symbol: {symbol}")
        return {}
    
    if start_date > end_date:
        print("ERROR: Start date must be before end date")
        return {}
    
    print(f"CABE ANALYSIS: {symbol} from {start_date} to {end_date}")
    print("=" * 80)
    print("NOTE: Polygon.io intraday data is only available for the most recent trading day")
    print("=" * 80)
    
    analysis_results = {
        'symbol': symbol,
        'date_range': {'start': start_date, 'end': end_date},
        'analysis_timestamp': datetime.now().isoformat(),
        'daily_data': [],
        'summary_stats': {},
        'event_analytics': {}
    }
    
    try:
        # Get daily data for the entire range
        daily_aggs = fetch_with_rate_limit_handling(
            client.list_aggs,
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=50000
        )
        
        if not daily_aggs:
            print(f"NO_DATA: No daily data available for {symbol}")
            return analysis_results
        
        # Process daily data efficiently
        daily_data = process_aggregation_data(daily_aggs)
        if not daily_data:
            return analysis_results
        
        # Build a date->close lookup for next-day analysis
        date_to_close = {timestamp.split()[0]: close for timestamp, close in zip(daily_data['timestamps'], daily_data['closes'])}
        date_list = [timestamp.split()[0] for timestamp in daily_data['timestamps']]
    
        # Filter trading days in the range efficiently
        trading_days = []
        for i, timestamp in enumerate(daily_data['timestamps']):
            date = timestamp.split()[0]  # Extract date part
            if start_date <= date <= end_date:
                trading_days.append({
                    'date': date,
                    'open': daily_data['opens'][i],
                    'close': daily_data['closes'][i],
                    'high': daily_data['highs'][i],
                    'low': daily_data['lows'][i],
                    'volume': daily_data['volumes'][i],
                    'index': i
                })
    
        print(f"FOUND: {len(trading_days)} trading days in the date range")
    
        # Analyze each trading day
        total_vwap_crosses = 0
        total_rsi_overbought_crosses = 0
        total_rsi_oversold_crosses = 0
        days_with_intraday = 0
        days_without_intraday = 0
    
        for i, day in enumerate(trading_days, 1):
            print(f"DAY {i}/{len(trading_days)}: {day['date']}")
            print("-" * 50)
        
            # Calculate daily metrics
            daily_change = day['close'] - day['open']
            daily_change_pct = (daily_change / day['open']) * 100
        
            print(f"OPEN: ${day['open']:.2f}")
            print(f"CLOSE: ${day['close']:.2f}")
            print(f"VOLUME: {day['volume']:,.0f} shares")
            print(f"HIGH: ${day['high']:.2f}")
            print(f"LOW: ${day['low']:.2f}")
            print(f"CHANGE: ${daily_change:+.2f} ({daily_change_pct:+.2f}%)")
        
            # Get intraday data
            intraday_data = analyze_intraday_data(symbol, day['date'])
        
            day_result = {
                'date': day['date'],
                'daily_metrics': {
                    'open': day['open'],
                    'close': day['close'],
                    'high': day['high'],
                    'low': day['low'],
                    'volume': day['volume'],
                    'change': daily_change,
                    'change_pct': daily_change_pct
                },
                'intraday_analysis': None
            }
        
            if intraday_data:
                days_with_intraday += 1
                print(f"VWAP: ${intraday_data.vwap:.2f}")
                print(f"VWAP_CROSSES: {len(intraday_data.crosses['vwap'])}")
            
                if intraday_data.rsi_values and intraday_data.rsi_values[-1]:
                    print(f"CURRENT_RSI: {intraday_data.rsi_values[-1]:.2f}")
            
                # Print cross information
                if intraday_data.crosses['rsi_overbought']:
                    print(f"RSI_OVERBOUGHT_CROSSES: {len(intraday_data.crosses['rsi_overbought'])}")
                    for cross in intraday_data.crosses['rsi_overbought']:
                        print(f"  {cross['timestamp']}: RSI {cross['direction']} 80 (RSI: {cross['value']:.2f})")
            
                if intraday_data.crosses['rsi_oversold']:
                    print(f"RSI_OVERSOLD_CROSSES: {len(intraday_data.crosses['rsi_oversold'])}")
                    for cross in intraday_data.crosses['rsi_oversold']:
                        print(f"  {cross['timestamp']}: RSI {cross['direction']} 20 (RSI: {cross['value']:.2f})")
            
                # Find next trading day's close
                this_idx = day['index']
                next_day_close = None
                if this_idx + 1 < len(date_list):
                    next_day = date_list[this_idx + 1]
                    next_day_close = date_to_close.get(next_day)
            
                # Annotate all event types with next day direction and return
                event_day_close = day['close']
                intraday_data.crosses['vwap'] = annotate_event_window_return(
                    intraday_data.crosses['vwap'], day['index'], date_list, date_to_close, window
                )
                intraday_data.crosses['rsi_overbought'] = annotate_event_window_return(
                    intraday_data.crosses['rsi_overbought'], day['index'], date_list, date_to_close, window
                )
                intraday_data.crosses['rsi_oversold'] = annotate_event_window_return(
                    intraday_data.crosses['rsi_oversold'], day['index'], date_list, date_to_close, window
                )
            
                total_vwap_crosses += len(intraday_data.crosses['vwap'])
                total_rsi_overbought_crosses += len(intraday_data.crosses['rsi_overbought'])
                total_rsi_oversold_crosses += len(intraday_data.crosses['rsi_oversold'])
            
                day_result['intraday_analysis'] = intraday_data.to_dict()
            
                # Create visualizations if requested
                if create_viz:
                    create_visualizations(intraday_data)
            else:
                days_without_intraday += 1
                print("NO_INTRADAY_DATA")
        
            analysis_results['daily_data'].append(day_result)
            print()
        
            # Adaptive rate limiting
            if i < len(trading_days):
                print("WAITING: 3 seconds before next API call...")
                time.sleep(3)
        
        # Calculate summary statistics
        analysis_results['summary_stats'] = {
            'total_trading_days': len(trading_days),
            'days_with_intraday_data': days_with_intraday,
            'days_without_intraday_data': days_without_intraday,
            'total_vwap_crosses': total_vwap_crosses,
            'total_rsi_overbought_crosses': total_rsi_overbought_crosses,
            'total_rsi_oversold_crosses': total_rsi_oversold_crosses,
            'avg_vwap_crosses_per_day': total_vwap_crosses / days_with_intraday if days_with_intraday > 0 else 0,
            'avg_rsi_overbought_crosses_per_day': total_rsi_overbought_crosses / days_with_intraday if days_with_intraday > 0 else 0,
            'avg_rsi_oversold_crosses_per_day': total_rsi_oversold_crosses / days_with_intraday if days_with_intraday > 0 else 0
        }
        
        # --- Signal Aggregation ---
        all_vwap_events = []
        all_rsi_overbought_events = []
        all_rsi_oversold_events = []
        for day in analysis_results['daily_data']:
            ia = day.get('intraday_analysis')
            if ia:
                all_vwap_events.extend(ia['crosses']['vwap'])
                all_rsi_overbought_events.extend(ia['crosses']['rsi_overbought'])
                all_rsi_oversold_events.extend(ia['crosses']['rsi_oversold'])
        analysis_results['event_analytics'] = {
            'vwap': aggregate_event_stats(all_vwap_events, window),
            'rsi_overbought': aggregate_event_stats(all_rsi_overbought_events, window),
            'rsi_oversold': aggregate_event_stats(all_rsi_oversold_events, window)
        }
        # Print event analytics summary
        print("SIGNAL AGGREGATION SUMMARY")
        for k, v in analysis_results['event_analytics'].items():
            print(f"{k.upper()}: count={v['count']}, win_rate={v['win_rate']}, avg_return={v['avg_return']}, median_return={v['median_return']}")
        
        # Print summary
        print("=" * 80)
        print("CABE SUMMARY STATISTICS")
        print("=" * 80)
        print(f"TOTAL_TRADING_DAYS: {len(trading_days)}")
        print(f"DAYS_WITH_INTRADAY_DATA: {days_with_intraday}")
        print(f"DAYS_WITHOUT_INTRADAY_DATA: {days_without_intraday}")
        print()
        
        if days_with_intraday > 0:
            print(f"TOTAL_VWAP_CROSSES: {total_vwap_crosses}")
            print(f"AVG_VWAP_CROSSES_PER_DAY: {total_vwap_crosses/days_with_intraday:.1f}")
            print(f"TOTAL_RSI_OVERBOUGHT_CROSSES: {total_rsi_overbought_crosses}")
            print(f"TOTAL_RSI_OVERSOLD_CROSSES: {total_rsi_oversold_crosses}")
            print(f"AVG_RSI_OVERBOUGHT_CROSSES_PER_DAY: {total_rsi_overbought_crosses/days_with_intraday:.1f}")
            print(f"AVG_RSI_OVERSOLD_CROSSES_PER_DAY: {total_rsi_oversold_crosses/days_with_intraday:.1f}")
    
        # Save data if requested
        if save_data:
            os.makedirs("cabe_output", exist_ok=True)
            filename = f"cabe_output/{symbol}_{start_date}_{end_date}_analysis.json"
            with open(filename, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            print(f"DATA_SAVED: {filename}")
        
        print("CABE ANALYSIS COMPLETE")
        return analysis_results
        
    except Exception as e:
        print(f"ERROR: Error during analysis: {e}")
        return analysis_results

def main():
    parser = argparse.ArgumentParser(description='CABE (Cool Ass Backtest Engine) - Professional Stock Analysis Tool')
    parser.add_argument('symbol', help='Stock symbol (e.g., SPY)')
    parser.add_argument('start_date', help='Start date in MMDDYY format (e.g., 061325)')
    parser.add_argument('end_date', help='End date in MMDDYY format (e.g., 061327)')
    parser.add_argument('--window', type=int, default=1, help='Event window size in trading days (default: 1)')
    parser.add_argument('--no-viz', action='store_true', help='Do not create visualizations')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital for trade simulation (default: 100000)')
    args = parser.parse_args()
    if not validate_symbol(args.symbol):
        print(f"ERROR: Invalid symbol format: {args.symbol}")
        exit(1)
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if not start_date or not end_date:
        exit(1)
    daily_aggs = fetch_with_rate_limit_handling(
        client.list_aggs,
        ticker=args.symbol,
        multiplier=1,
        timespan="day",
        from_=start_date,
        to=end_date,
        limit=50000
    )
    daily_data = process_aggregation_data(daily_aggs)
    if not daily_data:
        print(f"NO_DATA: No daily data available for {args.symbol}")
        return
    date_to_close = {datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'): close for ts, close in zip(daily_data['timestamps'], daily_data['closes'])}
    date_list = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') for ts in daily_data['timestamps']]
    trading_days = []
    for i, date in enumerate(date_list):
        if start_date <= date <= end_date:
            trading_days.append({'date': date, 'index': i})
    all_events = []
    for day in trading_days:
        aggs = fetch_with_rate_limit_handling(
            client.list_aggs,
            ticker=args.symbol,
            multiplier=5,
            timespan="minute",
            from_=day['date'],
            to=day['date'],
            limit=50000
        )
        if not aggs:
            continue
        data = process_aggregation_data(aggs)
        vwap = np.average((np.array(data['highs']) + np.array(data['lows']) + np.array(data['closes'])) / 3, weights=np.array(data['volumes']))
        # RSI calculation
        closes = pd.Series(data['closes'])
        deltas = closes.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        avg_gain = gains.ewm(span=14, adjust=False).mean()
        avg_loss = losses.ewm(span=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi_values = 100 - (100 / (1 + rs))
        df = pd.DataFrame({
            'timestamp': data['timestamps'],
            'open': data['opens'],
            'high': data['highs'],
            'low': data['lows'],
            'close': data['closes'],
            'volume': data['volumes'],
            'rsi': rsi_values
        })
        df['datetime'] = pd.to_datetime(df['timestamp'])
        vwap_events = detect_vwap_crosses(data['closes'], vwap, data['timestamps'])
        rsi_overbought_events = detect_rsi_crosses(rsi_values, 80, data['timestamps'])
        rsi_oversold_events = detect_rsi_crosses(rsi_values, 20, data['timestamps'])
        vwap_events = annotate_event_window_return(vwap_events, day['index'], date_list, date_to_close, args.window)
        rsi_overbought_events = annotate_event_window_return(rsi_overbought_events, day['index'], date_list, date_to_close, args.window)
        rsi_oversold_events = annotate_event_window_return(rsi_oversold_events, day['index'], date_list, date_to_close, args.window)
        for e in vwap_events:
            e['type'] = 'vwap'
            e['event_day'] = day['date']
            e['symbol'] = args.symbol
        for e in rsi_overbought_events:
            e['type'] = 'rsi_overbought'
            e['event_day'] = day['date']
            e['symbol'] = args.symbol
        for e in rsi_oversold_events:
            e['type'] = 'rsi_oversold'
            e['event_day'] = day['date']
            e['symbol'] = args.symbol
        all_events.extend(vwap_events + rsi_overbought_events + rsi_oversold_events)
        if not args.no_viz:
            create_visualizations(df, args.symbol, day['date'], vwap)
    if not all_events:
        print("NO_EVENTS: No events detected in the given range.")
        return
    events_df = pd.DataFrame(all_events)
    os.makedirs('cabe_output', exist_ok=True)
    events_df.to_csv(f'cabe_output/{args.symbol}_{start_date}_{end_date}_events.csv', index=False)
    print(f"CSV_SAVED: cabe_output/{args.symbol}_{start_date}_{end_date}_events.csv")
    for event_type in ['vwap', 'rsi_overbought', 'rsi_oversold']:
        stats = aggregate_event_stats(events_df[events_df['type'] == event_type].to_dict('records'), args.window)
        print(f"{event_type.upper()} - count: {stats['count']}, win_rate: {stats['win_rate']}, avg_return: {stats['avg_return']}, median_return: {stats['median_return']}")
    # --- Trade Simulation ---
    trades_df, trade_summary = simulate_trades(events_df.to_dict('records'), args.window, args.capital)
    trades_df.to_csv(f'cabe_output/{args.symbol}_{start_date}_{end_date}_trades.csv', index=False)
    print(f"TRADES_CSV_SAVED: cabe_output/{args.symbol}_{start_date}_{end_date}_trades.csv")
    print(f"TRADE SIMULATION: num_trades={trade_summary['num_trades']}, total_pnl={trade_summary['total_pnl']}, final_capital={trade_summary['final_capital']}, avg_trade_return={trade_summary['avg_trade_return']}, median_trade_return={trade_summary['median_trade_return']}")

if __name__ == "__main__":
    main()