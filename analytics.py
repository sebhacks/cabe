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

def annotate_event_window_return(events: List[Dict], event_day_idx: int, date_list: List[str], 
                               date_to_close: Dict[str, float], window: int) -> List[Dict]:
    """Annotate events with window return calculations"""
    try:
        if not events:
            return events
            
        if not isinstance(window, int) or window < 1:
            logger.error("Invalid window size")
            return events
            
        window_return = None
        if event_day_idx + window < len(date_list):
            future_date = date_list[event_day_idx + window]
            future_close = date_to_close.get(future_date)
            event_close = date_to_close.get(date_list[event_day_idx])
            
            if future_close is not None and event_close is not None and event_close != 0:
                window_return = (future_close - event_close) / event_close
                
        for event in events:
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