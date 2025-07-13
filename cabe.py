#!/usr/bin/env python3
"""
CABE (Cool Ass Backtest Engine) - Professional Quantitative Trading Analysis Tool
Features: VWAP, RSI, Bollinger Bands, Cross Detection, Data Visualization with Polygon.io API integration
"""

import os
import re
import json
import logging
from dotenv import load_dotenv
import time
from datetime import datetime, timezone, timedelta
import argparse
import pandas as pd
import numpy as np
from polygon import RESTClient
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from analytics import (
    detect_rsi_crosses, 
    detect_vwap_crosses, 
    annotate_event_window_return, 
    aggregate_event_stats, 
    simulate_trades,
    calculate_bollinger_bands,
    detect_bollinger_crosses
)

# Configure logging for better debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cabe.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load API key securely
load_dotenv()
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

if not POLYGON_API_KEY:
    logger.error('Polygon.io API key not found. Please set POLYGON_API_KEY in your .env file.')
    raise RuntimeError('Polygon.io API key not found. Please set POLYGON_API_KEY in your .env file.')

# Enhanced security: Validate API key format and length
if not re.match(r'^[A-Za-z0-9_-]{20,}$', POLYGON_API_KEY):
    logger.error('Invalid API key format detected')
    raise ValueError('Invalid API key format detected')

# Initialize Polygon client with timeout and retry configuration
client = RESTClient(api_key=POLYGON_API_KEY)

# Configure matplotlib for professional output with memory optimization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Security constants
MAX_SYMBOL_LENGTH = 10
MAX_DATE_RANGE_DAYS = 365  # Limit to 1 year to prevent excessive API usage
MIN_DATA_POINTS = 20
MAX_RETRIES = 3
RATE_LIMIT_WAIT = 5  # seconds

def validate_symbol(symbol: str) -> bool:
    """Enhanced symbol validation for security"""
    if not symbol or len(symbol) > MAX_SYMBOL_LENGTH:
        return False
    # More restrictive pattern to prevent injection attacks
    return bool(re.match(r'^[A-Z]{1,5}$', symbol.upper()))

def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range to prevent excessive API usage"""
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if end_dt < start_dt:
            logger.error("End date must be after start date")
            return False
            
        date_diff = (end_dt - start_dt).days
        if date_diff > MAX_DATE_RANGE_DAYS:
            logger.error(f"Date range too large ({date_diff} days). Maximum allowed: {MAX_DATE_RANGE_DAYS} days")
            return False
            
        return True
    except ValueError:
        logger.error("Invalid date format in validation")
        return False

def parse_date(date_str: str) -> Optional[str]:
    """Parse date string to YYYY-MM-DD format with enhanced validation"""
    try:
        if len(date_str) != 6:
            logger.error("Date must be in MMDDYY format (e.g., 061323 for June 13, 2023)")
            return None
        
        month = int(date_str[:2])
        day = int(date_str[2:4])
        year = int(date_str[4:6])
        
        # Enhanced date validation
        if not (1 <= month <= 12):
            logger.error("Invalid month value")
            return None
        if not (1 <= day <= 31):
            logger.error("Invalid day value")
            return None
            
        year += 2000 if year < 50 else 1900
        
        # Validate the actual date (handles leap years, etc.)
        parsed_date = datetime(year, month, day)
        
        # Check if date is not in the future
        if parsed_date > datetime.now():
            logger.error("Date cannot be in the future")
            return None
            
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return None

def process_aggregation_data(aggs: List) -> Optional[Dict[str, List]]:
    """Efficiently process aggregation data with memory optimization"""
    if not aggs:
        return None
    
    try:
        # Use more efficient numpy operations
        data = np.array([[agg.open, agg.high, agg.low, agg.close, agg.volume, agg.timestamp] for agg in aggs])
        
        # Validate data integrity
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.warning("Invalid data detected in aggregation")
            return None
        
        return {
            'opens': data[:, 0].tolist(),
            'highs': data[:, 1].tolist(),
            'lows': data[:, 2].tolist(),
            'closes': data[:, 3].tolist(),
            'volumes': data[:, 4].tolist(),
            'timestamps': [datetime.fromtimestamp(ts/1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S') for ts in data[:, 5]]
        }
    except Exception as e:
        logger.error(f"Error processing aggregation data: {e}")
        return None

def fetch_with_rate_limit_handling(fetch_fn, *args, **kwargs) -> List:
    """Handle Polygon API rate limits with enhanced error handling and logging"""
    for attempt in range(MAX_RETRIES):
        try:
            result = list(fetch_fn(*args, **kwargs))
            logger.info(f"Successfully fetched data (attempt {attempt + 1})")
            return result
        except Exception as e:
            error_str = str(e).lower()
            if 'rate limit' in error_str or '429' in error_str or 'too many requests' in error_str:
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{MAX_RETRIES}). Waiting {RATE_LIMIT_WAIT} seconds...")
                time.sleep(RATE_LIMIT_WAIT)
            elif 'unauthorized' in error_str or '401' in error_str:
                logger.error("API key is invalid or expired")
                raise
            elif 'not found' in error_str or '404' in error_str:
                logger.warning("Data not found for the specified parameters")
                return []
            else:
                logger.error(f"Unexpected error during API call: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error("Failed after multiple retries")
    return []

def create_visualizations(df: pd.DataFrame, symbol: str, date: str, vwap: float) -> None:
    """Generate and save professional data visualizations with memory optimization"""
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f'{symbol} Analysis for {date}', fontsize=16)

        # Plot price, VWAP, and Bollinger Bands
        ax1.plot(df['datetime'], df['close'], label='Close Price', color='blue', alpha=0.7)
        ax1.axhline(y=vwap, color='purple', linestyle='--', label=f'VWAP ({vwap:.2f})')
        
        if 'bb_upper' in df.columns:
            ax1.plot(df['datetime'], df['bb_middle'], color='orange', linestyle='--', alpha=0.7, label='BB Middle')
            ax1.plot(df['datetime'], df['bb_upper'], color='red', linestyle='--', alpha=0.5, label='BB Upper')
            ax1.plot(df['datetime'], df['bb_lower'], color='green', linestyle='--', alpha=0.5, label='BB Lower')
            ax1.fill_between(df['datetime'], df['bb_lower'], df['bb_upper'], color='grey', alpha=0.1)

        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True)

        # Plot RSI
        ax2.plot(df['datetime'], df['rsi'], label='RSI', color='green')
        ax2.axhline(80, color='red', linestyle='--', alpha=0.5, label='Overbought (80)')
        ax2.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Oversold (20)')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True)

        # Plot Volume
        ax3.bar(df['datetime'], df['volume'], label='Volume', color='black', alpha=0.5)
        ax3.set_ylabel('Volume')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Ensure output directory exists
        os.makedirs('cabe_output', exist_ok=True)
        filename = f"cabe_output/{symbol}_{date.replace('-', '')}_chart.png"
        
        # Save with optimized settings
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()  # Explicitly close to free memory
        
        logger.info(f"Visualization saved: {filename}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")

def main():
    """CABE main function with enhanced error handling and security"""
    parser = argparse.ArgumentParser(description='CABE - Professional Stock Analysis Tool')
    parser.add_argument('symbol', help='Stock symbol (e.g., SPY)')
    parser.add_argument('start_date', help='Start date in MMDDYY format')
    parser.add_argument('end_date', help='End date in MMDDYY format')
    parser.add_argument('--window', type=int, default=1, help='Event window size in days (default: 1)')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualizations')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital for simulation')
    args = parser.parse_args()

    # Enhanced input validation
    if not validate_symbol(args.symbol):
        logger.error(f"Invalid symbol: {args.symbol}")
        return

    # Validate window size
    if args.window < 1 or args.window > 30:
        logger.error("Window size must be between 1 and 30 days")
        return

    # Validate capital
    if args.capital <= 0:
        logger.error("Initial capital must be positive")
        return

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if not start_date or not end_date:
        return

    # Validate date range
    if not validate_date_range(start_date, end_date):
        return

    logger.info(f"Starting analysis for {args.symbol} from {start_date} to {end_date}")

    try:
        # Fetch daily data
        daily_aggs = fetch_with_rate_limit_handling(
            client.list_aggs,
            ticker=args.symbol, multiplier=1, timespan="day",
            from_=start_date, to=end_date, limit=50000
        )
        daily_data = process_aggregation_data(daily_aggs)
        if not daily_data:
            logger.error(f"No daily data for {args.symbol}")
            return

        # Process daily data more efficiently
        date_to_close = {}
        date_list = []
        for ts, close in zip(daily_data['timestamps'], daily_data['closes']):
            date = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
            date_to_close[date] = close
            date_list.append(date)
        
        trading_days = [{'date': date, 'index': i} for i, date in enumerate(date_list) if start_date <= date <= end_date]
        all_events = []

        logger.info(f"Processing {len(trading_days)} trading days")

        for day in trading_days:
            try:
                aggs = fetch_with_rate_limit_handling(
                    client.list_aggs,
                    ticker=args.symbol, multiplier=5, timespan="minute",
                    from_=day['date'], to=day['date'], limit=50000
                )
                if not aggs: 
                    logger.warning(f"No intraday data for {day['date']}")
                    continue
                
                data = process_aggregation_data(aggs)
                if not data or len(data['closes']) < MIN_DATA_POINTS: 
                    logger.warning(f"Insufficient data for {day['date']} (need {MIN_DATA_POINTS}, got {len(data['closes']) if data else 0})")
                    continue

                # --- Calculations with error handling ---
                try:
                    vwap = np.average((np.array(data['highs']) + np.array(data['lows']) + np.array(data['closes'])) / 3, weights=np.array(data['volumes']))
                    
                    closes_s = pd.Series(data['closes'])
                    rsi_values = 100 - (100 / (1 + (closes_s.diff().where(closes_s.diff() > 0, 0).ewm(span=14).mean() / -closes_s.diff().where(closes_s.diff() < 0, 0).ewm(span=14).mean())))
                    
                    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(data['closes'])

                    df = pd.DataFrame({
                        'timestamp': data['timestamps'], 'open': data['opens'], 'high': data['highs'], 
                        'low': data['lows'], 'close': data['closes'], 'volume': data['volumes'],
                        'rsi': rsi_values, 'bb_middle': bb_middle, 'bb_upper': bb_upper, 'bb_lower': bb_lower
                    })
                    df['datetime'] = pd.to_datetime(df['timestamp'])

                    # --- Event Detection ---
                    events = {
                        'vwap': detect_vwap_crosses(data['closes'], vwap, data['timestamps']),
                        'rsi_overbought': detect_rsi_crosses(rsi_values, 80, data['timestamps']),
                        'rsi_oversold': detect_rsi_crosses(rsi_values, 20, data['timestamps']),
                        'bollinger': detect_bollinger_crosses(data['closes'], bb_upper, bb_lower, data['timestamps'], data['volumes'])
                    }

                    for event_type, event_list in events.items():
                        annotated_events = annotate_event_window_return(event_list, day['index'], date_list, date_to_close, args.window)
                        for e in annotated_events:
                            e.update({'type': event_type, 'event_day': day['date'], 'symbol': args.symbol})
                            all_events.append(e)

                    if not args.no_viz:
                        create_visualizations(df, args.symbol, day['date'], vwap)

                except Exception as e:
                    logger.error(f"Error processing day {day['date']}: {e}")
                    continue

            except Exception as e:
                logger.error(f"Error fetching data for {day['date']}: {e}")
                continue

        if not all_events:
            logger.info("No events detected.")
            return

        # --- Reporting & Saving with error handling ---
        try:
            events_df = pd.DataFrame(all_events)
            os.makedirs('cabe_output', exist_ok=True)
            events_filename = f'cabe_output/{args.symbol}_{start_date}_{end_date}_events.csv'
            events_df.to_csv(events_filename, index=False)
            logger.info(f"Events saved: {events_filename}")

            for event_type in ['vwap', 'rsi_overbought', 'rsi_oversold', 'bollinger']:
                stats = aggregate_event_stats(events_df[events_df['type'] == event_type].to_dict('records'), args.window)
                if stats['count'] > 0:
                    print(f"{event_type.upper():<16} - count: {stats['count']:<5} win_rate: {stats['win_rate']:.2%} avg_return: {stats['avg_return']:.4%} median_return: {stats['median_return']:.4%}")
                else:
                    print(f"{event_type.upper():<16} - count: 0")

            # --- Trade Simulation ---
            trades_df, trade_summary = simulate_trades(events_df.to_dict('records'), args.window, args.capital)
            trades_filename = f'cabe_output/{args.symbol}_{start_date}_{end_date}_trades.csv'
            trades_df.to_csv(trades_filename, index=False)
            logger.info(f"Trades saved: {trades_filename}")
            print(f"TRADE_SIM: num_trades={trade_summary['num_trades']}, pnl={trade_summary['total_pnl']:.2f}, final_capital={trade_summary['final_capital']:.2f}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()