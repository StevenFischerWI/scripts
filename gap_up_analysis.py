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

def find_gap_ups(conn, date: str, gap_threshold: float = 0.05) -> pd.DataFrame:
    """
    Find all gap-ups over the threshold for a given date.
    Gap-up = (open - previous_close) / previous_close > threshold
    """
    logging.debug(f"Finding gap-ups for {date} with threshold {gap_threshold*100}%")
    query = """
    SELECT
        c.symbol,
        c.open,
        c.high,
        c.low,
        c.close,
        c.date,
        COALESCE(c.volume, 0) as volume,
        prev.close as prev_close,
        (c.open - prev.close) / prev.close as gap_percent
    FROM candle c
    JOIN LATERAL (
        SELECT close
        FROM candle p
        WHERE p.symbol = c.symbol AND p.date < %s
        ORDER BY p.date DESC
        LIMIT 1
    ) prev ON TRUE
    WHERE c.date = %s
    AND (c.open - prev.close) / prev.close >= %s
    ORDER BY gap_percent DESC;
    """
    
    df = pd.read_sql_query(query, conn, params=(date, date, gap_threshold))
    logging.debug(f"Found {len(df)} gap-ups for {date}")
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
    Get the distance from 330-period simple moving average as percentage of close price.
    Returns the percentage distance (positive if above SMA, negative if below).
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

def analyze_gap_ups_for_date(conn, date: str, csv_writer) -> Dict:
    """Analyze gap-ups for a specific date and write results to CSV."""
    logging.info(f"Analyzing gap-ups for {date}...")
    
    gap_ups = find_gap_ups(conn, date)
    
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
    SELECT symbol, date, high, low, close, COALESCE(volume, 0) AS volume
    FROM candle
    WHERE date BETWEEN %s AND %s
      AND symbol = ANY(%s)
    """
    vwap_df = pd.read_sql_query(vwap_query, conn, params=(start_date, date, symbols))
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

    # 10-day window after the gap day for retracement and MFE/MAE
    post_query = """
    SELECT symbol, date, high, low
    FROM candle
    WHERE date > %s AND date <= %s
      AND symbol = ANY(%s)
    ORDER BY symbol, date
    """
    post_df = pd.read_sql_query(post_query, conn, params=(date, end_date_10, symbols))
    post_df['date'] = pd.to_datetime(post_df['date']) if not post_df.empty else post_df

    # Gap day prices mapped by symbol
    open_by_symbol = dict(zip(gap_ups['symbol'], gap_ups['open']))
    close_by_symbol = dict(zip(gap_ups['symbol'], gap_ups['close']))

    # Compute per-symbol retracement and MFE/MAE
    retraced_map = {}
    mfe_mae_map = {}
    lowest_low_map = {}
    if not post_df.empty:
        for sym, g in post_df.groupby('symbol'):
            lowest_low = g['low'].min()
            highest_high = g['high'].max()
            lowest_low_map[sym] = lowest_low

            avwap = anchored_vwap_map.get(sym, np.nan)
            retraced = False
            days_to_retrace = None
            if pd.notna(avwap):
                hit = g[g['low'] <= avwap]
                if not hit.empty:
                    retraced = True
                    first_hit_date = hit['date'].iloc[0]
                    days_to_retrace = (first_hit_date - gap_date_dt).days

            close_price = close_by_symbol.get(sym, np.nan)
            if pd.notna(close_price) and close_price > 0:
                mfe_dollars = highest_high - close_price
                mae_dollars = close_price - lowest_low
                mfe_percent = (mfe_dollars / close_price) * 100
                mae_percent = (mae_dollars / close_price) * 100
            else:
                mfe_dollars = mae_dollars = mfe_percent = mae_percent = 0.0

            retraced_map[sym] = (retraced, days_to_retrace)
            mfe_mae_map[sym] = (mfe_dollars, mae_dollars, mfe_percent, mae_percent)
    else:
        for sym in symbols:
            lowest_low_map[sym] = np.nan
            retraced_map[sym] = (False, None)
            mfe_mae_map[sym] = (0.0, 0.0, 0.0, 0.0)

    # 330-day SMA distance for all symbols
    sma_query = """
    SELECT symbol, close
    FROM (
        SELECT symbol, close, date,
               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
        FROM candle
        WHERE symbol = ANY(%s) AND date <= %s
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
    for _, row in gap_ups.iterrows():
        symbol = row['symbol']
        gap_percent = row['gap_percent']
        open_price = row['open']
        close_price = row['close']

        anchored_vwap = float(anchored_vwap_map.get(symbol, 0.0))
        retraced, days_to_retrace = retraced_map.get(symbol, (False, None))
        lowest_low = lowest_low_map.get(symbol, np.nan)
        mfe_dollars, mae_dollars, mfe_percent, mae_percent = mfe_mae_map.get(symbol, (0.0, 0.0, 0.0, 0.0))

        sma_base = sma_330_map.get(symbol)
        sma_330_distance = ((close_price - sma_base) / sma_base) * 100 if sma_base and sma_base > 0 else None

        if retraced:
            retraced_count += 1

        # Retracement percentage using lowest low in the 10-day window
        if anchored_vwap > 0 and open_price > anchored_vwap and pd.notna(lowest_low):
            max_possible_retrace = open_price - anchored_vwap
            actual_retrace = open_price - float(lowest_low)
            retrace_percentage = (actual_retrace / max_possible_retrace) * 100 if max_possible_retrace > 0 else 0.0
        else:
            retrace_percentage = 0.0

        csv_rows.append({
            'ticker': symbol,
            'date': date,
            'gap_up_percent': round(gap_percent * 100, 2),
            'open_price': round(open_price, 2),
            'close_price': round(close_price, 2),
            'anchored_vwap': round(anchored_vwap, 2),
            'retraced_to_vwap': retraced,
            'days_to_retrace': days_to_retrace if days_to_retrace is not None else '',
            'retrace_percentage': round(retrace_percentage, 2),
            'mfe_dollars': round(mfe_dollars, 2),
            'mae_dollars': round(mae_dollars, 2),
            'mfe_percent': round(mfe_percent, 2),
            'mae_percent': round(mae_percent, 2),
            'sma_330_distance_percent': round(sma_330_distance, 2) if sma_330_distance is not None else '',
            'spy_daily_gain_percent': round(spy_daily_gain, 2) if spy_daily_gain is not None else ''
        })

        results.append({
            'symbol': symbol,
            'gap_percent': gap_percent * 100,  # Convert to percentage
            'open_price': open_price,
            'close_price': close_price,
            'anchored_vwap': anchored_vwap,
            'retraced_to_vwap': retraced,
            'days_to_retrace': days_to_retrace,
            'retrace_percentage': retrace_percentage,
            'mfe_dollars': mfe_dollars,
            'mae_dollars': mae_dollars,
            'mfe_percent': mfe_percent,
            'mae_percent': mae_percent,
            'sma_330_distance_percent': sma_330_distance,
            'spy_daily_gain_percent': spy_daily_gain
        })

    # Write all rows at once for this date
    if csv_rows:
        csv_writer.writerows(csv_rows)
    
    total_gap_ups = len(results)
    retracement_rate = (retraced_count / total_gap_ups * 100) if total_gap_ups > 0 else 0
    
    return {
        'date': date,
        'total_gap_ups': total_gap_ups,
        'retraced_to_vwap': retraced_count,
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
        csv_filename = 'output.csv'
        logging.info(f"Writing detailed results to {csv_filename}")
        
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['ticker', 'date', 'gap_up_percent', 'open_price', 'close_price', 'anchored_vwap', 'retraced_to_vwap', 'days_to_retrace', 'retrace_percentage', 'mfe_dollars', 'mae_dollars', 'mfe_percent', 'mae_percent', 'sma_330_distance_percent', 'spy_daily_gain_percent']
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            
            for i, date in enumerate(dates, 1):
                logging.info(f"Processing date {i}/{len(dates)}: {date}")
                result = analyze_gap_ups_for_date(conn, date, csv_writer)
                all_results.append(result)
                
                # Print summary for this date
                print(f"Date: {result['date']}")
                print(f"Total gap-ups (>=5%): {result['total_gap_ups']}")
                print(f"Retraced to VWAP: {result['retraced_to_vwap']}")
                print(f"Retracement rate: {result['retracement_rate']:.1f}%")
                
                # Print top gap-ups for this date
                if result['details']:
                    print("\nTop gap-ups:")
                    for detail in sorted(result['details'], key=lambda x: x['gap_percent'], reverse=True)[:5]:
                        retraced_text = "✓" if detail['retraced_to_vwap'] else "✗"
                        days_text = f" ({detail['days_to_retrace']} days)" if detail['days_to_retrace'] is not None else ""
                        print(f"  {detail['symbol']}: {detail['gap_percent']:.1f}% gap, "
                              f"Anchored VWAP: ${detail['anchored_vwap']:.2f}, "
                              f"Retraced: {retraced_text}{days_text}, "
                              f"Retrace %: {detail['retrace_percentage']:.1f}%, "
                              f"MFE: ${detail['mfe_dollars']:.2f} ({detail['mfe_percent']:.1f}%), "
                              f"MAE: ${detail['mae_dollars']:.2f} ({detail['mae_percent']:.1f}%)")
                
                print("-" * 40)
        
        # Overall summary
        total_gap_ups = sum(r['total_gap_ups'] for r in all_results)
        total_retraced = sum(r['retraced_to_vwap'] for r in all_results)
        overall_rate = (total_retraced / total_gap_ups * 100) if total_gap_ups > 0 else 0
        
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"Total gap-ups analyzed: {total_gap_ups}")
        print(f"Total that retraced to VWAP: {total_retraced}")
        print(f"Overall retracement rate: {overall_rate:.1f}%")
        print(f"\nDetailed results saved to: output.csv")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        conn.dispose()

if __name__ == "__main__":
    main()
