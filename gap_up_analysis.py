#!/usr/bin/env python3
"""
Gap-up analysis script for ZenBot Scanner database.

Finds all gap-ups over 3% each day and determines how many retrace to VWAP 
from 3 days before earnings.
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import argparse
import logging
import csv
from typing import List, Dict, Tuple, Optional
from sqlalchemy import create_engine

# Database connection parameters
DB_CONFIG = {
    'host': 'zenbot',
    'database': 'zenbot-scanner',
    'user': 'zenbot',
    'password': 'zenbot'
}

def connect_to_db():
    """Connect to PostgreSQL database using SQLAlchemy."""
    try:
        logging.info("Connecting to database...")
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string)
        logging.info("Database connection established")
        return engine
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        return None

def calculate_vwap(df: pd.DataFrame) -> float:
    """Calculate Volume Weighted Average Price (VWAP) for a DataFrame of candles."""
    if df.empty:
        return 0.0
    
    # Typical price = (high + low + close) / 3
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # VWAP = sum(typical_price * volume) / sum(volume)
    if 'volume' in df.columns and df['volume'].sum() > 0:
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
    else:
        # If no volume data, use simple average of typical prices
        vwap = typical_price.mean()
    
    return vwap

def find_gap_ups(conn, date: str, gap_threshold: float = 0.05, gap_mode: str = 'up') -> pd.DataFrame:
    """
    Find all gap-ups over the threshold for a given date.
    Gap-up = (open - previous_close) / previous_close > threshold
    """
    logging.debug(f"Finding gap-{gap_mode}s for {date} with threshold {gap_threshold*100}%")
    if gap_mode == 'up':
        query = """
        SELECT
            c.symbol,
            c.open,
            c.high,
            c.low,
            c.close,
            c.date,
            c.volume,
            prev.close AS prev_close,
            (c.open - prev.close) / prev.close AS gap_percent
        FROM (
            SELECT DISTINCT ON (symbol, date)
                symbol,
                open,
                high,
                low,
                close,
                date,
                COALESCE(volume, 0) AS volume
            FROM candle
            WHERE date = %s
            ORDER BY symbol, date, COALESCE(volume, 0) DESC
        ) c
        JOIN LATERAL (
            SELECT close
            FROM candle p
            WHERE p.symbol = c.symbol AND p.date < %s
            ORDER BY p.date DESC
            LIMIT 1
        ) prev ON TRUE
        WHERE prev.close BETWEEN %s AND %s
          AND (c.open - prev.close) / prev.close >= %s
        ORDER BY gap_percent DESC;
        """
        params = (date, date, 20.0, 1000.0, gap_threshold)
    else:
        query = """
        SELECT
            c.symbol,
            c.open,
            c.high,
            c.low,
            c.close,
            c.date,
            c.volume,
            prev.close AS prev_close,
            (prev.close - c.open) / prev.close AS gap_percent
        FROM (
            SELECT DISTINCT ON (symbol, date)
                symbol,
                open,
                high,
                low,
                close,
                date,
                COALESCE(volume, 0) AS volume
            FROM candle
            WHERE date = %s
            ORDER BY symbol, date, COALESCE(volume, 0) DESC
        ) c
        JOIN LATERAL (
            SELECT close
            FROM candle p
            WHERE p.symbol = c.symbol AND p.date < %s
            ORDER BY p.date DESC
            LIMIT 1
        ) prev ON TRUE
        WHERE prev.close BETWEEN %s AND %s
          AND (prev.close - c.open) / prev.close >= %s
        ORDER BY gap_percent DESC;
        """
        params = (date, date, 20.0, 1000.0, gap_threshold)
    
    df = pd.read_sql_query(query, conn, params=params)
    df = df.drop_duplicates(subset=['symbol', 'date'], keep='first')
    logging.debug(f"Found {len(df)} gap-{gap_mode}s for {date}")
    return df

def get_anchored_vwap_through_gap_day(conn, symbol: str, gap_date: str) -> float:
    """Get anchored VWAP from 3 days before gap-up through close of gap-up day."""
    start_date = (datetime.strptime(gap_date, '%Y-%m-%d') - timedelta(days=3)).strftime('%Y-%m-%d')
    
    query = """
    SELECT high, low, close, COALESCE(volume, 0) as volume
    FROM candle 
    WHERE symbol = %s 
    AND date >= %s 
    AND date <= %s
    ORDER BY date
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, start_date, gap_date))
    
    if df.empty:
        return 0.0
    
    return calculate_vwap(df)

def check_retracement_to_vwap(conn, symbol: str, gap_date: str, vwap_price: float, open_price: float, days_to_check: int = 10) -> Tuple[bool, Optional[int], float]:
    """
    Check if the stock retraced to the VWAP price within the specified number of days.
    Returns (retraced, days_to_retrace, retrace_percentage)
    """
    end_date = (datetime.strptime(gap_date, '%Y-%m-%d') + timedelta(days=days_to_check)).strftime('%Y-%m-%d')
    
    query = """
    SELECT date, low
    FROM candle 
    WHERE symbol = %s 
    AND date > %s 
    AND date <= %s
    ORDER BY date
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, gap_date, end_date))
    
    if df.empty:
        return False, None, 0.0
    
    # Find the lowest price reached during the period
    lowest_price = df['low'].min()
    
    # Calculate retracement percentage: how much it moved from open toward VWAP
    # If open > vwap, retracement = (open - lowest) / (open - vwap) * 100
    # Allow values > 100% if it retraced below VWAP
    if open_price > vwap_price:
        max_possible_retrace = open_price - vwap_price
        actual_retrace = open_price - lowest_price
        retrace_percentage = (actual_retrace / max_possible_retrace) * 100 if max_possible_retrace > 0 else 0.0
    else:
        retrace_percentage = 0.0
    
    # Check each day to find when it first retraced to VWAP
    gap_date_dt = datetime.strptime(gap_date, '%Y-%m-%d')
    
    for _, row in df.iterrows():
        if row['low'] <= vwap_price:
            retrace_date = pd.to_datetime(row['date']).to_pydatetime()
            days_to_retrace = (retrace_date - gap_date_dt).days
            return True, days_to_retrace, retrace_percentage
    
    return False, None, retrace_percentage

def get_mfe_mae_10_days(conn, symbol: str, gap_date: str, close_price: float) -> Tuple[float, float, float, float]:
    """
    Get Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE) 
    for 10 days following the gap-up.
    Returns (mfe_dollars, mae_dollars, mfe_percent, mae_percent)
    """
    end_date = (datetime.strptime(gap_date, '%Y-%m-%d') + timedelta(days=10)).strftime('%Y-%m-%d')
    
    query = """
    SELECT high, low
    FROM candle 
    WHERE symbol = %s 
    AND date > %s 
    AND date <= %s
    ORDER BY date
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, gap_date, end_date))
    
    if df.empty:
        return 0.0, 0.0, 0.0, 0.0
    
    # Find highest high (MFE) and lowest low (MAE) during the 10-day period
    highest_price = df['high'].max()
    lowest_price = df['low'].min()
    
    # Calculate MFE and MAE in dollars and percentages
    mfe_dollars = highest_price - close_price
    mae_dollars = close_price - lowest_price
    
    mfe_percent = (mfe_dollars / close_price) * 100 if close_price > 0 else 0.0
    mae_percent = (mae_dollars / close_price) * 100 if close_price > 0 else 0.0
    
    return mfe_dollars, mae_dollars, mfe_percent, mae_percent

def get_sma_330_distance(conn, symbol: str, gap_date: str, close_price: float) -> float:
    """
    Compute the percent difference between the gap-day close and the 330-day Simple Moving Average (SMA).

    Definition:
        sma_330_distance_percent = 100 * (gap_day_close - SMA_330) / SMA_330

    Interpretation:
        Positive => close above SMA_330
        Negative => close below SMA_330
    """
    # Get 330 days of data before the gap date (plus some buffer for weekends/holidays)
    start_date = (datetime.strptime(gap_date, '%Y-%m-%d') - timedelta(days=500)).strftime('%Y-%m-%d')
    
    query = """
    SELECT close, date
    FROM candle 
    WHERE symbol = %s 
    AND date <= %s
    AND date >= %s
    ORDER BY date DESC
    LIMIT 330
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, gap_date, start_date))
    
    if len(df) < 330:
        return None  # Not enough data for 330-period SMA
    
    # Calculate 330-period simple moving average
    sma_330 = df['close'].mean()
    
    # Calculate percentage distance from SMA
    if sma_330 > 0:
        distance_percent = ((close_price - sma_330) / sma_330) * 100
        return distance_percent
    
    return None

def get_spy_daily_gain(conn, gap_date: str) -> float:
    """
    Get SPY's percentage gain on the gap date compared to previous close.
    Returns the percentage gain (positive for up, negative for down).
    """
    query = """
    WITH spy_data AS (
        SELECT 
            date,
            close,
            LAG(close) OVER (ORDER BY date) as prev_close
        FROM candle 
        WHERE symbol = 'SPY'
        AND date <= %s
        ORDER BY date DESC
        LIMIT 2
    )
    SELECT 
        (close - prev_close) / prev_close * 100 as daily_gain_percent
    FROM spy_data
    WHERE date = %s
    AND prev_close IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn, params=(gap_date, gap_date))
    
    if df.empty:
        return None
    
    return df['daily_gain_percent'].iloc[0]

def analyze_gap_ups_for_date(conn, date: str, csv_writer, gap_mode: str = 'up') -> Dict:
    """Analyze gap-ups for a specific date and write results to CSV."""
    logging.info(f"Analyzing gap-{gap_mode}s for {date}...")
    
    gap_ups = find_gap_ups(conn, date, gap_mode=gap_mode)
    
    if gap_ups.empty:
        logging.info(f"No gap-ups found for {date}")
        return {
            'date': date,
            'total_gap_ups': 0,
            'retraced_to_vwap': 0,
            'retracement_rate': 0.0,
            'details': []
        }
    
    results = []
    retraced_count = 0
    
    logging.info(f"Processing {len(gap_ups)} gap-ups for {date}")
    # Batch computations for performance (anchored VWAP, retracement/MFE/MAE, SMA-330)
    symbols = gap_ups['symbol'].tolist()
    gap_date_dt = datetime.strptime(date, '%Y-%m-%d')
    start_date = (gap_date_dt - timedelta(days=3)).strftime('%Y-%m-%d')
    end_date_10 = (gap_date_dt + timedelta(days=10)).strftime('%Y-%m-%d')

    # Anchored VWAP from 3 days before through gap day for all symbols
    vwap_query = """
    SELECT symbol, date, high, low, close, volume
    FROM (
        SELECT DISTINCT ON (symbol, date)
            symbol, date, high, low, close, COALESCE(volume, 0) AS volume
        FROM candle
        WHERE date BETWEEN %s AND %s
          AND symbol = ANY(%s)
        ORDER BY symbol, date, COALESCE(volume, 0) DESC
    ) d
    """
    vwap_df = pd.read_sql_query(vwap_query, conn, params=(start_date, date, symbols))
    vwap_df = vwap_df.drop_duplicates(subset=['symbol', 'date'], keep='first')
    anchored_vwap_map = {}
    if not vwap_df.empty:
        vwap_df['typical'] = (vwap_df['high'] + vwap_df['low'] + vwap_df['close']) / 3
        vwap_df['tp_x_vol'] = vwap_df['typical'] * vwap_df['volume']
        agg = vwap_df.groupby('symbol', as_index=False).agg(
            tpv=('tp_x_vol', 'sum'),
            vol=('volume', 'sum'),
            typical_mean=('typical', 'mean')
        )
        agg['anchored_vwap'] = np.where(agg['vol'] > 0, agg['tpv'] / agg['vol'], agg['typical_mean'])
        anchored_vwap_map = dict(zip(agg['symbol'], agg['anchored_vwap']))

    # Filter candidates by AVWAP "room" >= 2% of prior close, aligned with gap mode
    if not gap_ups.empty:
        avwap_series = gap_ups['symbol'].map(anchored_vwap_map).astype(float)
        threshold = 0.02 * gap_ups['prev_close']

        if gap_mode == 'up':
            # Short fade to AVWAP on gap-ups: open above AVWAP with sufficient room
            cond = avwap_series.notna() & (gap_ups['open'] > avwap_series) & ((gap_ups['open'] - avwap_series) >= threshold)
        else:
            # Long bounce to AVWAP on gap-downs: open below AVWAP with sufficient room
            cond = avwap_series.notna() & (gap_ups['open'] < avwap_series) & ((avwap_series - gap_ups['open']) >= threshold)

        gap_ups = gap_ups[cond].reset_index(drop=True)
        symbols = gap_ups['symbol'].tolist()
        if gap_ups.empty:
            logging.info(f"No gap-{gap_mode}s meet AVWAP room >= 2% of prior close for {date}")
            return {
                'date': date,
                'total_gap_ups': 0,
                'retraced_to_vwap': 0,
                'retracement_rate': 0.0,
                'details': []
            }

    # Next 10 TRADING days after the gap day (rn = 1..10) for retracement and MFE/MAE
    post_query = """
    WITH base AS (
        SELECT DISTINCT ON (symbol, date)
            symbol, date, high, low, COALESCE(volume, 0) AS volume
        FROM candle
        WHERE date > %s
          AND symbol = ANY(%s)
        ORDER BY symbol, date, COALESCE(volume, 0) DESC
    ),
    next_days AS (
        SELECT
            symbol, date, high, low,
            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date ASC) AS rn
        FROM base
    )
    SELECT symbol, date, high, low, rn
    FROM next_days
    WHERE rn <= 10
    ORDER BY symbol, date
    """
    post_df = pd.read_sql_query(post_query, conn, params=(date, symbols))
    if not post_df.empty:
        post_df['date'] = pd.to_datetime(post_df['date'])

    # Gap day prices mapped by symbol
    open_by_symbol = dict(zip(gap_ups['symbol'], gap_ups['open']))
    close_by_symbol = dict(zip(gap_ups['symbol'], gap_ups['close']))

    # Compute per-symbol retracement and MFE/MAE for 5 and 10 trading days
    retraced5_map, retraced10_map = {}, {}
    days5_map, days10_map = {}, {}
    ll5_map, ll10_map, hh5_map, hh10_map = {}, {}, {}, {}
    mfe_mae_5_map, mfe_mae_10_map = {}, {}

    if not post_df.empty:
        for sym, g in post_df.groupby('symbol'):
            g5 = g[g['rn'] <= 5]
            g10 = g[g['rn'] <= 10]

            ll5 = g5['low'].min() if not g5.empty else np.nan
            ll10 = g10['low'].min() if not g10.empty else np.nan
            hh5 = g5['high'].max() if not g5.empty else np.nan
            hh10 = g10['high'].max() if not g10.empty else np.nan

            ll5_map[sym], ll10_map[sym] = ll5, ll10
            hh5_map[sym], hh10_map[sym] = hh5, hh10

            avwap = anchored_vwap_map.get(sym, np.nan)

            if pd.notna(avwap):
                if gap_mode == 'up':
                    hit5 = g5[g5['low'] <= avwap]
                    hit10 = g10[g10['low'] <= avwap]
                else:
                    hit5 = g5[g5['high'] >= avwap]
                    hit10 = g10[g10['high'] >= avwap]
            else:
                hit5 = hit10 = pd.DataFrame()

            retraced5 = not hit5.empty
            retraced10 = not hit10.empty
            retraced5_map[sym], retraced10_map[sym] = retraced5, retraced10
            days5_map[sym] = int(hit5['rn'].iloc[0]) if retraced5 else None
            days10_map[sym] = int(hit10['rn'].iloc[0]) if retraced10 else None

            # MFE/MAE vs gap-day close
            close_price = close_by_symbol.get(sym, np.nan)
            if pd.notna(close_price) and close_price > 0:
                mfe5 = (hh5 - close_price) if pd.notna(hh5) else 0.0
                mae5 = (close_price - ll5) if pd.notna(ll5) else 0.0
                mfe10 = (hh10 - close_price) if pd.notna(hh10) else 0.0
                mae10 = (close_price - ll10) if pd.notna(ll10) else 0.0

                mfe5_pct = (mfe5 / close_price) * 100
                mae5_pct = (mae5 / close_price) * 100
                mfe10_pct = (mfe10 / close_price) * 100
                mae10_pct = (mae10 / close_price) * 100
            else:
                mfe5 = mae5 = mfe10 = mae10 = 0.0
                mfe5_pct = mae5_pct = mfe10_pct = mae10_pct = 0.0

            mfe_mae_5_map[sym] = (mfe5, mae5, mfe5_pct, mae5_pct)
            mfe_mae_10_map[sym] = (mfe10, mae10, mfe10_pct, mae10_pct)
    else:
        for sym in symbols:
            ll5_map[sym] = ll10_map[sym] = np.nan
            hh5_map[sym] = hh10_map[sym] = np.nan
            retraced5_map[sym] = retraced10_map[sym] = False
            days5_map[sym] = days10_map[sym] = None
            mfe_mae_5_map[sym] = mfe_mae_10_map[sym] = (0.0, 0.0, 0.0, 0.0)

    # 330-day SMA distance for all symbols
    sma_query = """
    SELECT symbol, close
    FROM (
        SELECT symbol, close, date,
               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
        FROM (
            SELECT DISTINCT ON (symbol, date)
                symbol, close, date
            FROM candle
            WHERE symbol = ANY(%s) AND date <= %s
            ORDER BY symbol, date DESC
        ) d
    ) t
    WHERE rn <= 330
    """
    sma_df = pd.read_sql_query(sma_query, conn, params=(symbols, date))
    sma_330_map = {}
    if not sma_df.empty:
        sma_330_map = sma_df.groupby('symbol')['close'].mean().to_dict()

    # SPY daily gain for this date (single query)
    spy_daily_gain = get_spy_daily_gain(conn, date)

    # Build results and CSV rows
    csv_rows = []
    retraced_count_5 = 0
    retraced_count_10 = 0
    for _, row in gap_ups.iterrows():
        symbol = row['symbol']
        gap_percent = row['gap_percent']
        open_price = row['open']
        close_price = row['close']

        anchored_vwap = float(anchored_vwap_map.get(symbol, 0.0))
        sma_base = sma_330_map.get(symbol)
        sma_330_distance = ((close_price - sma_base) / sma_base) * 100 if sma_base and sma_base > 0 else None

        # 5d and 10d retracement info
        retraced5 = retraced5_map.get(symbol, False)
        days5 = days5_map.get(symbol)
        retraced10 = retraced10_map.get(symbol, False)
        days10 = days10_map.get(symbol)

        if retraced5:
            retraced_count_5 += 1
        if retraced10:
            retraced_count_10 += 1

        # Retracement percentage for 5d and 10d
        retrace_pct_5 = 0.0
        retrace_pct_10 = 0.0
        if anchored_vwap > 0:
            if gap_mode == 'up' and open_price > anchored_vwap:
                ll5 = ll5_map.get(symbol, np.nan)
                ll10 = ll10_map.get(symbol, np.nan)
                max_pos = open_price - anchored_vwap
                if pd.notna(ll5) and max_pos > 0:
                    retrace_pct_5 = (max(0.0, open_price - float(ll5)) / max_pos) * 100
                if pd.notna(ll10) and max_pos > 0:
                    retrace_pct_10 = (max(0.0, open_price - float(ll10)) / max_pos) * 100
            elif gap_mode == 'down' and open_price < anchored_vwap:
                hh5 = hh5_map.get(symbol, np.nan)
                hh10 = hh10_map.get(symbol, np.nan)
                max_pos = anchored_vwap - open_price
                if pd.notna(hh5) and max_pos > 0:
                    retrace_pct_5 = (max(0.0, float(hh5) - open_price) / max_pos) * 100
                if pd.notna(hh10) and max_pos > 0:
                    retrace_pct_10 = (max(0.0, float(hh10) - open_price) / max_pos) * 100

        # MFE/MAE maps
        mfe5, mae5, mfe5p, mae5p = mfe_mae_5_map.get(symbol, (0.0, 0.0, 0.0, 0.0))
        mfe10, mae10, mfe10p, mae10p = mfe_mae_10_map.get(symbol, (0.0, 0.0, 0.0, 0.0))

        csv_rows.append({
            'ticker': symbol,
            'date': date,
            'gap_up_percent': round(gap_percent * 100, 2),
            'open_price': round(open_price, 2),
            'close_price': round(close_price, 2),
            'anchored_vwap': round(anchored_vwap, 2),
            'retraced_to_vwap_5d': retraced5,
            'days_to_retrace_5d': days5 if days5 is not None else '',
            'retrace_percentage_5d': round(retrace_pct_5, 2),
            'mfe_dollars_5d': round(mfe5, 2),
            'mae_dollars_5d': round(mae5, 2),
            'mfe_percent_5d': round(mfe5p, 2),
            'mae_percent_5d': round(mae5p, 2),
            'retraced_to_vwap_10d': retraced10,
            'days_to_retrace_10d': days10 if days10 is not None else '',
            'retrace_percentage_10d': round(retrace_pct_10, 2),
            'mfe_dollars_10d': round(mfe10, 2),
            'mae_dollars_10d': round(mae10, 2),
            'mfe_percent_10d': round(mfe10p, 2),
            'mae_percent_10d': round(mae10p, 2),
            'sma_330_distance_percent': round(sma_330_distance, 2) if sma_330_distance is not None else '',
            'spy_daily_gain_percent': round(spy_daily_gain, 2) if spy_daily_gain is not None else ''
        })

        results.append({
            'symbol': symbol,
            'gap_percent': gap_percent * 100,  # Convert to percentage
            'open_price': open_price,
            'close_price': close_price,
            'anchored_vwap': anchored_vwap,
            'retraced_to_vwap_5d': retraced5,
            'days_to_retrace_5d': days5,
            'retrace_percentage_5d': retrace_pct_5,
            'mfe_dollars_5d': mfe5,
            'mae_dollars_5d': mae5,
            'mfe_percent_5d': mfe5p,
            'mae_percent_5d': mae5p,
            'retraced_to_vwap': retraced10,
            'days_to_retrace': days10,
            'retrace_percentage': retrace_pct_10,
            'mfe_dollars': mfe10,
            'mae_dollars': mae10,
            'mfe_percent': mfe10p,
            'mae_percent': mae10p,
            'sma_330_distance_percent': sma_330_distance,
            'spy_daily_gain_percent': spy_daily_gain
        })

    # Write all rows at once for this date
    if csv_rows:
        csv_writer.writerows(csv_rows)
    
    total_gap_ups = len(results)
    retracement_rate_5d = (retraced_count_5 / total_gap_ups * 100) if total_gap_ups > 0 else 0
    retracement_rate = (retraced_count_10 / total_gap_ups * 100) if total_gap_ups > 0 else 0
    
    return {
        'date': date,
        'total_gap_ups': total_gap_ups,
        'retraced_to_vwap_5d': retraced_count_5,
        'retracement_rate_5d': retracement_rate_5d,
        'retraced_to_vwap': retraced_count_10,
        'retracement_rate': retracement_rate,
        'details': results
    }

def get_available_dates(conn, start_date: str = None, end_date: str = None) -> List[str]:
    """Get list of available trading dates from the database between start and end dates."""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    query = """
    SELECT DISTINCT date 
    FROM candle 
    WHERE date >= %s AND date <= %s
    ORDER BY date ASC
    """
    
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    # Convert date column to datetime if it's not already
    df['date'] = pd.to_datetime(df['date'])
    return df['date'].dt.strftime('%Y-%m-%d').tolist()

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze gap-ups between start and end dates')
    parser.add_argument('--start-date', '-s', type=str, 
                       help='Start date for analysis (YYYY-MM-DD). Defaults to 30 days ago.')
    parser.add_argument('--end-date', '-e', type=str,
                       help='End date for analysis (YYYY-MM-DD). Defaults to today.')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--gap-mode', choices=['up', 'down'], default='up',
                       help='Gap direction to analyze: up (default) or down')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        # Get trading dates between start and end dates
        logging.info("Fetching available trading dates...")
        dates = get_available_dates(conn, start_date=args.start_date, end_date=args.end_date)
        
        if not dates:
            logging.error("No trading dates found in the database for the specified period.")
            return
        
        start_display = args.start_date or "30 days ago"
        end_display = args.end_date or "today"
        logging.info(f"Analyzing {len(dates)} trading dates from {start_display} to {end_display}")
        print("=" * 80)
        
        all_results = []
        
        # Create CSV file for detailed results
        direction_label = 'short' if args.gap_mode == 'up' else 'long'
        csv_filename = f'output-{direction_label}.csv'
        logging.info(f"Writing detailed results to {csv_filename}")
        
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = [
                'ticker','date','gap_up_percent','open_price','close_price','anchored_vwap',
                'retraced_to_vwap_5d','days_to_retrace_5d','retrace_percentage_5d','mfe_dollars_5d','mae_dollars_5d','mfe_percent_5d','mae_percent_5d',
                'retraced_to_vwap_10d','days_to_retrace_10d','retrace_percentage_10d','mfe_dollars_10d','mae_dollars_10d','mfe_percent_10d','mae_percent_10d',
                'sma_330_distance_percent','spy_daily_gain_percent'
            ]
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            
            for i, date in enumerate(dates, 1):
                logging.info(f"Processing date {i}/{len(dates)}: {date}")
                result = analyze_gap_ups_for_date(conn, date, csv_writer, gap_mode=args.gap_mode)
                all_results.append(result)
                
                # Print summary for this date
                print(f"Date: {result['date']}")
                print(f"Total gap-{args.gap_mode}s (>=5%): {result['total_gap_ups']}")
                print(f"Retraced to VWAP (5d): {result['retraced_to_vwap_5d']}  rate: {result['retracement_rate_5d']:.1f}%")
                print(f"Retraced to VWAP (10d): {result['retraced_to_vwap']}  rate: {result['retracement_rate']:.1f}%")
                
                # Print top gap-ups for this date
                if result['details']:
                    print(f"\nTop gap-{args.gap_mode}s:")
                    for detail in sorted(result['details'], key=lambda x: x['gap_percent'], reverse=True)[:5]:
                        r5 = "✓" if detail['retraced_to_vwap_5d'] else "✗"
                        d5 = f" ({detail['days_to_retrace_5d']}d)" if detail['days_to_retrace_5d'] is not None else ""
                        r10 = "✓" if detail['retraced_to_vwap'] else "✗"
                        d10 = f" ({detail['days_to_retrace']}d)" if detail['days_to_retrace'] is not None else ""
                        print(f"  {detail['symbol']}: {detail['gap_percent']:.1f}% gap, AVWAP ${detail['anchored_vwap']:.2f}, "
                              f"5d: {r5}{d5}, {detail['retrace_percentage_5d']:.1f}% | "
                              f"10d: {r10}{d10}, {detail['retrace_percentage']:.1f}%, "
                              f"MFE 5d: ${detail['mfe_dollars_5d']:.2f} ({detail['mfe_percent_5d']:.1f}%), "
                              f"MAE 5d: ${detail['mae_dollars_5d']:.2f} ({detail['mae_percent_5d']:.1f}%), "
                              f"MFE 10d: ${detail['mfe_dollars']:.2f} ({detail['mfe_percent']:.1f}%), "
                              f"MAE 10d: ${detail['mae_dollars']:.2f} ({detail['mae_percent']:.1f}%)")
                
                print("-" * 40)
        
        # Overall summary
        total_gap_ups = sum(r['total_gap_ups'] for r in all_results)
        total_retraced_5d = sum(r.get('retraced_to_vwap_5d', 0) for r in all_results)
        total_retraced_10d = sum(r['retraced_to_vwap'] for r in all_results)
        overall_rate_5d = (total_retraced_5d / total_gap_ups * 100) if total_gap_ups > 0 else 0
        overall_rate_10d = (total_retraced_10d / total_gap_ups * 100) if total_gap_ups > 0 else 0
        
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"Total gap-{args.gap_mode}s analyzed: {total_gap_ups}")
        print(f"Total that retraced to VWAP (5d): {total_retraced_5d}  rate: {overall_rate_5d:.1f}%")
        print(f"Total that retraced to VWAP (10d): {total_retraced_10d}  rate: {overall_rate_10d:.1f}%")
        print(f"\nDetailed results saved to: {csv_filename}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        conn.dispose()

if __name__ == "__main__":
    main()
