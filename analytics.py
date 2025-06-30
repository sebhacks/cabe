import numpy as np
import pandas as pd

def detect_rsi_crosses(rsi_values, threshold, timestamps):
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

def detect_vwap_crosses(closes, vwap, timestamps):
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

def annotate_event_window_return(events, event_day_idx, date_list, date_to_close, window):
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

def aggregate_event_stats(event_list, window=1):
    df = pd.DataFrame(event_list)
    count = len(df)
    win_rate = (df[f'window_{window}_return'] > 0).mean() if count > 0 else None
    avg_return = df[f'window_{window}_return'].mean() if count > 0 else None
    median_return = df[f'window_{window}_return'].median() if count > 0 else None
    return {
        'count': count,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'median_return': median_return
    }

def simulate_trades(event_list, window=1, initial_capital=100000):
    """
    Simulate buying on each event and selling after N days. Returns per-trade and cumulative P&L.
    """
    df = pd.DataFrame(event_list)
    trades = []
    capital = initial_capital
    for idx, row in df.iterrows():
        ret = row.get(f'window_{window}_return')
        if ret is not None:
            trade_pnl = capital * ret
            trades.append({
                'event_type': row.get('type'),
                'event_day': row.get('event_day'),
                'symbol': row.get('symbol'),
                'return': ret,
                'pnl': trade_pnl
            })
            capital += trade_pnl
    trades_df = pd.DataFrame(trades)
    summary = {
        'num_trades': len(trades_df),
        'total_pnl': trades_df['pnl'].sum() if not trades_df.empty else 0,
        'final_capital': capital,
        'avg_trade_return': trades_df['return'].mean() if not trades_df.empty else None,
        'median_trade_return': trades_df['return'].median() if not trades_df.empty else None
    }
    return trades_df, summary 