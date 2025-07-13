import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

def validate_input_data(data: List, min_length: int = 2) -> bool:
    """Validate input data for analysis functions"""
    if not data or len(data) < min_length:
        return False
    if not all(isinstance(x, (int, float, np.number)) for x in data):
        return False
    return True

def detect_rsi_crosses(rsi_values: List[float], threshold: float, timestamps: List[str]) -> List[Dict]:
    """Detect RSI crosses with enhanced validation and error handling"""
    try:
        if not validate_input_data(rsi_values, 2):
            logger.warning("Insufficient RSI data for cross detection")
            return []
            
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 100:
            logger.error("Invalid RSI threshold value")
            return []
            
        rsi_array = np.array(rsi_values)
        valid_mask = ~pd.isna(rsi_array)
        valid_rsi = rsi_array[valid_mask]
        valid_timestamps = [timestamps[i] for i, valid in enumerate(valid_mask) if valid]
        
        crosses = []
        if len(valid_rsi) < 2:
            return crosses
            
        above = valid_rsi > threshold
        for i in range(1, len(valid_rsi)):
            if above[i] != above[i-1]:
                crosses.append({
                    'timestamp': valid_timestamps[i],
                    'value': float(valid_rsi[i]),
                    'direction': 'above' if above[i] else 'below',
                    'threshold': threshold
                })
        return crosses
        
    except Exception as e:
        logger.error(f"Error in RSI cross detection: {e}")
        return []

def detect_vwap_crosses(closes: List[float], vwap: float, timestamps: List[str]) -> List[Dict]:
    """Detect VWAP crosses with enhanced validation"""
    try:
        if not validate_input_data(closes, 2):
            logger.warning("Insufficient price data for VWAP cross detection")
            return []
            
        if not isinstance(vwap, (int, float)) or vwap <= 0:
            logger.error("Invalid VWAP value")
            return []
            
        closes_array = np.array(closes)
        above = closes_array > vwap
        crosses = []
        
        for i in range(1, len(closes_array)):
            if above[i] != above[i-1]:
                crosses.append({
                    'timestamp': timestamps[i],
                    'value': float(closes_array[i]),
                    'direction': 'above' if above[i] else 'below',
                    'threshold': vwap
                })
        return crosses
        
    except Exception as e:
        logger.error(f"Error in VWAP cross detection: {e}")
        return []

def calculate_trade_return(event: Dict, event_day_idx: int, df: pd.DataFrame, date_list: List[str], 
                         date_to_close: Dict[str, float], event_type: str = None) -> float:
    """
    Calculate trade return based on sophisticated trading rules:
    
    VWAP Strategy:
    - If price crosses above VWAP: buy and hold until close of trading day OR until price crosses below VWAP
    - If price crosses below VWAP: sell short and hold until close OR until price crosses above VWAP
    
    Bollinger Bands Strategy:
    - If price crosses above Bollinger middle into upper band: buy and hold until close OR until price crosses below Bollinger middle
    - If price crosses below Bollinger middle: sell short and hold until close OR until price crosses above Bollinger middle
    
    RSI Strategy:
    - If RSI enters oversold region (<20): buy and hold until close of next day OR until RSI reaches overbought (>80)
    - If RSI enters overbought region (>80): sell short and hold until close of next day OR until RSI reaches oversold (<20)
    """
    try:
        # Use passed event_type if provided, otherwise try to get from event
        event_type = event_type or event.get('type')
        direction = event.get('direction')
        event_date = date_list[event_day_idx]
        
        # Find the VWAP value (should be constant for the analysis period)
        vwap = df['vwap'].iloc[0] if 'vwap' in df.columns else None
        
        # Initialize trade variables
        entry_price = date_to_close.get(event_date)
        if entry_price is None or entry_price == 0:
            return None
            
        exit_price = None
        exit_reason = None
        
        # Process based on event type
        if event_type == 'vwap':
            if direction == 'above':
                # Buy signal: hold until close or cross below VWAP
                exit_price, exit_reason = _find_vwap_exit(df, event_day_idx, vwap, 'below', date_to_close, date_list)
            elif direction == 'below':
                # Sell short signal: hold until close or cross above VWAP
                exit_price, exit_reason = _find_vwap_exit(df, event_day_idx, vwap, 'above', date_to_close, date_list)
                # For short trades, we need to reverse the return calculation
                if exit_price is not None:
                    return (entry_price - exit_price) / entry_price
                    
        elif event_type == 'bollinger':
            if direction == 'above_upper':
                # Buy signal: hold until close or cross below Bollinger middle
                exit_price, exit_reason = _find_bollinger_exit(df, event_day_idx, 'below_middle', date_to_close, date_list)
            elif direction == 'below_lower':
                # Sell short signal: hold until close or cross above Bollinger middle
                exit_price, exit_reason = _find_bollinger_exit(df, event_day_idx, 'above_middle', date_to_close, date_list)
                # For short trades, reverse the return calculation
                if exit_price is not None:
                    return (entry_price - exit_price) / entry_price
                    
        elif event_type in ['rsi_overbought', 'rsi_oversold']:
            if event_type == 'rsi_oversold':
                # Buy signal: hold until close of next day or RSI reaches overbought
                exit_price, exit_reason = _find_rsi_exit(df, event_day_idx, 'overbought', date_to_close, date_list)
            elif event_type == 'rsi_overbought':
                # Sell short signal: hold until close of next day or RSI reaches oversold
                exit_price, exit_reason = _find_rsi_exit(df, event_day_idx, 'oversold', date_to_close, date_list)
                # For short trades, reverse the return calculation
                if exit_price is not None:
                    return (entry_price - exit_price) / entry_price
        
        # Calculate return for long positions
        if exit_price is not None and exit_price != 0:
            return (exit_price - entry_price) / entry_price
            
        return None
        
    except Exception as e:
        logger.error(f"Error calculating trade return: {e}")
        return None

def _find_vwap_exit(df: pd.DataFrame, start_idx: int, vwap: float, exit_direction: str, 
                   date_to_close: Dict[str, float], date_list: List[str]) -> Tuple[float, str]:
    """Find exit price and reason for VWAP trades"""
    try:
        # Check if we can exit on the same day
        if start_idx < len(df) - 1:
            for idx in range(start_idx + 1, len(df)):
                row = df.iloc[idx]
                prev_row = df.iloc[idx - 1]
                
                # Check for cross in the opposite direction
                prev_above = prev_row['close'] > vwap
                curr_above = row['close'] > vwap
                
                if exit_direction == 'below' and prev_above and not curr_above:
                    return date_to_close.get(date_list[idx]), f"Crossed below VWAP on {date_list[idx]}"
                elif exit_direction == 'above' and not prev_above and curr_above:
                    return date_to_close.get(date_list[idx]), f"Crossed above VWAP on {date_list[idx]}"
        
        # If no cross found, exit at close of trading day
        if start_idx < len(df) - 1:
            return date_to_close.get(date_list[start_idx + 1]), f"Close of trading day {date_list[start_idx + 1]}"
        
        return None, "No exit found"
        
    except Exception as e:
        logger.error(f"Error finding VWAP exit: {e}")
        return None, "Error"

def _find_bollinger_exit(df: pd.DataFrame, start_idx: int, exit_direction: str, 
                        date_to_close: Dict[str, float], date_list: List[str]) -> Tuple[float, str]:
    """Find exit price and reason for Bollinger Band trades"""
    try:
        # Check if we can exit on the same day
        if start_idx < len(df) - 1:
            for idx in range(start_idx + 1, len(df)):
                row = df.iloc[idx]
                prev_row = df.iloc[idx - 1]
                
                # Check for cross of Bollinger middle band
                prev_above_middle = prev_row['close'] > prev_row['bb_middle']
                curr_above_middle = row['close'] > row['bb_middle']
                
                if exit_direction == 'below_middle' and prev_above_middle and not curr_above_middle:
                    return date_to_close.get(date_list[idx]), f"Crossed below Bollinger middle on {date_list[idx]}"
                elif exit_direction == 'above_middle' and not prev_above_middle and curr_above_middle:
                    return date_to_close.get(date_list[idx]), f"Crossed above Bollinger middle on {date_list[idx]}"
        
        # If no cross found, exit at close of trading day
        if start_idx < len(df) - 1:
            return date_to_close.get(date_list[start_idx + 1]), f"Close of trading day {date_list[start_idx + 1]}"
        
        return None, "No exit found"
        
    except Exception as e:
        logger.error(f"Error finding Bollinger exit: {e}")
        return None, "Error"

def _find_rsi_exit(df: pd.DataFrame, start_idx: int, exit_condition: str, 
                  date_to_close: Dict[str, float], date_list: List[str]) -> Tuple[float, str]:
    """Find exit price and reason for RSI trades"""
    try:
        # RSI trades exit at close of next day OR when RSI reaches opposite condition
        if start_idx < len(df) - 1:
            for idx in range(start_idx + 1, len(df)):
                row = df.iloc[idx]
                
                # Check for RSI condition
                if exit_condition == 'overbought' and row['rsi'] > 80:
                    return date_to_close.get(date_list[idx]), f"RSI reached overbought on {date_list[idx]}"
                elif exit_condition == 'oversold' and row['rsi'] < 20:
                    return date_to_close.get(date_list[idx]), f"RSI reached oversold on {date_list[idx]}"
        
        # If no RSI condition met, exit at close of next day
        if start_idx < len(df) - 1:
            return date_to_close.get(date_list[start_idx + 1]), f"Close of next day {date_list[start_idx + 1]}"
        
        return None, "No exit found"
        
    except Exception as e:
        logger.error(f"Error finding RSI exit: {e}")
        return None, "Error"

def annotate_event_window_return(events: List[Dict], event_day_idx: int, date_list: List[str], 
                               date_to_close: Dict[str, float], window: int, df: pd.DataFrame, event_type: str = None) -> List[Dict]:
    """
    Annotate events with sophisticated trading return calculations based on specific rules:
    
    VWAP Strategy:
    - Buy when price crosses above VWAP, sell when crosses below VWAP or at close
    - Short when price crosses below VWAP, cover when crosses above VWAP or at close
    
    Bollinger Bands Strategy:
    - Buy when price crosses above middle band into upper band, sell when crosses below middle
    - Short when price crosses below middle band, cover when crosses above middle
    
    RSI Strategy:
    - Buy when RSI enters oversold (<20), sell at close of next day or when RSI reaches overbought (>80)
    - Short when RSI enters overbought (>80), cover at close of next day or when RSI reaches oversold (<20)
    """
    try:
        if not events:
            return events
            
        for event in events:
            # Calculate return using sophisticated trading rules
            trade_return = calculate_trade_return(event, event_day_idx, df, date_list, date_to_close, event_type)
            event['trade_return'] = trade_return
            
            # Keep the old window return for backward compatibility
            if not isinstance(window, int) or window < 1:
                event[f'window_{window}_return'] = None
            else:
                window_return = None
                if event_day_idx + window < len(date_list):
                    future_date = date_list[event_day_idx + window]
                    future_close = date_to_close.get(future_date)
                    event_close = date_to_close.get(date_list[event_day_idx])
                    
                    if future_close is not None and event_close is not None and event_close != 0:
                        window_return = (future_close - event_close) / event_close
                        
                event[f'window_{window}_return'] = window_return
            
        return events
        
    except Exception as e:
        logger.error(f"Error in event window return annotation: {e}")
        return events

def aggregate_event_stats(event_list: List[Dict], window: int = 1) -> Dict:
    """Aggregate event statistics with enhanced error handling"""
    try:
        if not event_list:
            return {
                'count': 0,
                'win_rate': None,
                'avg_return': None,
                'median_return': None
            }
            
        df = pd.DataFrame(event_list)
        count = len(df)
        
        if count == 0:
            return {
                'count': 0,
                'win_rate': None,
                'avg_return': None,
                'median_return': None
            }
            
        return_col = f'window_{window}_return'
        if return_col not in df.columns:
            logger.warning(f"Return column {return_col} not found in event data")
            return {
                'count': count,
                'win_rate': None,
                'avg_return': None,
                'median_return': None
            }
            
        # Filter out None/NaN values for calculations
        valid_returns = df[return_col].dropna()
        
        win_rate = (valid_returns > 0).mean() if len(valid_returns) > 0 else None
        avg_return = valid_returns.mean() if len(valid_returns) > 0 else None
        median_return = valid_returns.median() if len(valid_returns) > 0 else None
        
        return {
            'count': count,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'median_return': median_return
        }
        
    except Exception as e:
        logger.error(f"Error in event stats aggregation: {e}")
        return {
            'count': 0,
            'win_rate': None,
            'avg_return': None,
            'median_return': None
        }

def simulate_trades(event_list: List[Dict], window: int = 1, initial_capital: float = 100000) -> Tuple[pd.DataFrame, Dict]:
    """Simulate trades with enhanced validation and error handling"""
    try:
        if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
            logger.error("Invalid initial capital value")
            return pd.DataFrame(), {
                'num_trades': 0,
                'total_pnl': 0,
                'final_capital': initial_capital,
                'avg_trade_return': None,
                'median_trade_return': None
            }
            
        if not event_list:
            return pd.DataFrame(), {
                'num_trades': 0,
                'total_pnl': 0,
                'final_capital': initial_capital,
                'avg_trade_return': None,
                'median_trade_return': None
            }
            
        df = pd.DataFrame(event_list)
        trades = []
        capital = initial_capital
        
        for idx, row in df.iterrows():
            try:
                # Use sophisticated trade_return if available, otherwise fall back to window return
                ret = row.get('trade_return')
                if ret is None or pd.isna(ret):
                    ret = row.get(f'window_{window}_return')
                
                if ret is not None and not pd.isna(ret):
                    trade_pnl = capital * ret
                    trades.append({
                        'event_type': row.get('type'),
                        'event_day': row.get('event_day'),
                        'symbol': row.get('symbol'),
                        'return': ret,
                        'pnl': trade_pnl
                    })
                    capital += trade_pnl
            except Exception as e:
                logger.warning(f"Error processing trade {idx}: {e}")
                continue
                
        trades_df = pd.DataFrame(trades)
        
        summary = {
            'num_trades': len(trades_df),
            'total_pnl': trades_df['pnl'].sum() if not trades_df.empty else 0,
            'final_capital': capital,
            'avg_trade_return': trades_df['return'].mean() if not trades_df.empty else None,
            'median_trade_return': trades_df['return'].median() if not trades_df.empty else None
        }
        
        return trades_df, summary
        
    except Exception as e:
        logger.error(f"Error in trade simulation: {e}")
        return pd.DataFrame(), {
            'num_trades': 0,
            'total_pnl': 0,
            'final_capital': initial_capital,
            'avg_trade_return': None,
            'median_trade_return': None
        }

def calculate_bollinger_bands(prices: List[float], window: int = 20, num_std_dev: float = 2) -> Tuple[List[float], List[float], List[float]]:
    """Calculate Bollinger Bands with enhanced validation"""
    try:
        if not validate_input_data(prices, window + 1):
            logger.warning("Insufficient price data for Bollinger Bands calculation")
            return [], [], []
            
        if not isinstance(window, int) or window < 2:
            logger.error("Invalid window size for Bollinger Bands")
            return [], [], []
            
        if not isinstance(num_std_dev, (int, float)) or num_std_dev <= 0:
            logger.error("Invalid standard deviation multiplier")
            return [], [], []
            
        prices_series = pd.Series(prices)
        rolling_mean = prices_series.rolling(window=window).mean()
        rolling_std = prices_series.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)
        
        return rolling_mean.tolist(), upper_band.tolist(), lower_band.tolist()
        
    except Exception as e:
        logger.error(f"Error in Bollinger Bands calculation: {e}")
        return [], [], []

def detect_bollinger_crosses(prices: List[float], upper_bands: List[float], lower_bands: List[float], 
                           timestamps: List[str], volumes: List[float], volume_factor: float = 1.5) -> List[Dict]:
    """Detect Bollinger Band crosses with volume confirmation"""
    try:
        if not validate_input_data(prices, 2):
            logger.warning("Insufficient price data for Bollinger cross detection")
            return []
            
        if not validate_input_data(upper_bands, 2) or not validate_input_data(lower_bands, 2):
            logger.warning("Insufficient Bollinger Band data")
            return []
            
        if not isinstance(volume_factor, (int, float)) or volume_factor <= 0:
            logger.error("Invalid volume factor")
            return []
            
        prices_array = np.array(prices)
        upper_bands_array = np.array(upper_bands)
        lower_bands_array = np.array(lower_bands)
        volumes_array = np.array(volumes)

        # Calculate average volume, ignoring initial period where it's NaN
        valid_volumes = volumes_array[:len(prices_array) - 20]
        if len(valid_volumes) == 0:
            logger.warning("Insufficient volume data for analysis")
            return []
            
        avg_volume = np.nanmean(valid_volumes)
        if np.isnan(avg_volume) or avg_volume <= 0:
            logger.warning("Invalid average volume calculation")
            return []

        crosses = []

        for i in range(1, len(prices_array)):
            # Skip if bands are not calculated yet (NaN)
            if pd.isna(upper_bands_array[i]) or pd.isna(lower_bands_array[i]):
                continue

            is_high_volume = volumes_array[i] > (avg_volume * volume_factor)

            # Cross above upper band
            if (prices_array[i-1] <= upper_bands_array[i-1] and 
                prices_array[i] > upper_bands_array[i] and is_high_volume):
                crosses.append({
                    'timestamp': timestamps[i],
                    'value': float(prices_array[i]),
                    'direction': 'above_upper',
                    'threshold': float(upper_bands_array[i]),
                    'volume_confirmed': True
                })

            # Cross below lower band
            if (prices_array[i-1] >= lower_bands_array[i-1] and 
                prices_array[i] < lower_bands_array[i] and is_high_volume):
                crosses.append({
                    'timestamp': timestamps[i],
                    'value': float(prices_array[i]),
                    'direction': 'below_lower',
                    'threshold': float(lower_bands_array[i]),
                    'volume_confirmed': True
                })
                
        return crosses
        
    except Exception as e:
        logger.error(f"Error in Bollinger cross detection: {e}")
        return [] 