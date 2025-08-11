#!/usr/bin/env python3
"""
Generate a self-contained HTML report from output-short.csv or output-long.csv.

Usage:
  python report_from_csv.py --csv output-short.csv --out report-short.html

Dependencies:
  pip install pandas plotly jinja2
"""
import argparse
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import plotly.express as px
from jinja2 import Template


TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <style>
    body{font-family:system-ui,Segoe UI,Arial;margin:16px;line-height:1.4}
    .section{margin-bottom:32px;}
    table{border-collapse:collapse}
    td,th{border:1px solid #ccc;padding:6px 8px}
    .meta{color:#555}
    .index{background:#f5f5f5;padding:16px;margin:16px 0;border-radius:4px}
    .index ul{margin:0;padding-left:20px}
    .index li{margin:4px 0}
    .perf-table .avwap-group{background-color:#e8f4fd;}
    .perf-table .gap-group{background-color:#fff2cc;}
    .perf-table .ema-group{background-color:#f0f8e8;}
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <p class="meta">Rows: {{ nrows }}, Dates: {{ ndates }}, Direction: {{ direction }}</p>

  <div class="index">
    <h2>Table of Contents</h2>
    <ul>
      <li><a href="#performance">Performance summary</a></li>
      <li><a href="#monthly-win">Monthly win rates</a></li>
      <li><a href="#monthly-comparison">Monthly win rate (5d vs 10d)</a></li>
      <li><a href="#gap-distribution">Gap size distribution</a></li>
      <li><a href="#retracement-bucket">Retracement rate by gap bucket (10d)</a></li>
      <li><a href="#win-gap">Win rate vs gap size</a></li>
      <li><a href="#win-avwap">Win rate vs distance to AVWAP</a></li>
      <li><a href="#win-21ema-above-below">Win rate: above vs below 21-EMA</a></li>
      <li><a href="#win-50ema-above-below">Win rate: above vs below 50-EMA</a></li>
      <li><a href="#win-100ema-above-below">Win rate: above vs below 100-EMA</a></li>
      <li><a href="#win-200ema-above-below">Win rate: above vs below 200-EMA</a></li>
      <li><a href="#win-avwap-current-above-below">Win rate: above vs below Current Quarter AVWAP</a></li>
      <li><a href="#win-avwap-2q-above-below">Win rate: above vs below 2Q Ago AVWAP</a></li>
      <li><a href="#win-avwap-3q-above-below">Win rate: above vs below 3Q Ago AVWAP</a></li>
      <li><a href="#win-avwap-1y-above-below">Win rate: above vs below 1Y Ago AVWAP</a></li>
      <li><a href="#win-sma330-above-below">Win rate: above vs below 330-SMA</a></li>
      <li><a href="#win-sma330-bucket">Win rate by distance to 330-SMA</a></li>
      <li><a href="#win-spy">Win rate vs SPY daily change</a></li>
      <li><a href="#equity-curve">Expected P&L (10k/trade) — Equity Curve</a></li>
      <li><a href="#gap-retrace-scatter">Gap size vs 10d retrace %</a></li>
      <li><a href="#spy-retrace">SPY vs 10d retrace rate (by day)</a></li>
      <li><a href="#top-mfe">Top 20 by 10d MFE%</a></li>
    </ul>
  </div>

  <div class="section" id="performance">
    <h2>Performance summary</h2>
    <p>AVWAP retrace rate: 5d: {{ overall_win_rate_5d }}% | 10d: {{ overall_win_rate_10d }}%</p>
    <p>21 EMA retrace rate: 5d: {{ ema_21_retrace_rate_5d }}% &nbsp;|&nbsp; 10d: {{ ema_21_retrace_rate_10d }}%</p>
    <p>Gap closure rate: 5d: {{ gap_close_rate_5d }}% &nbsp;|&nbsp; 10d: {{ gap_close_rate_10d }}%</p>
    <h3>5-day performance by year</h3>
    <div class="table-container">{{ perf_table_5|safe }}</div>
    <h3>10-day performance by year</h3>
    <div class="table-container">{{ perf_table_10|safe }}</div>
    <script>
    // Add column group styling after tables are rendered
    document.addEventListener('DOMContentLoaded', function() {
        const tables = document.querySelectorAll('.perf-table');
        tables.forEach(table => {
            const headers = table.querySelectorAll('th');
            const rows = table.querySelectorAll('tbody tr');
            
            headers.forEach((header, index) => {
                const text = header.textContent.trim();
                let className = '';
                if (text.includes('AVWAP')) className = 'avwap-group';
                else if (text.includes('Gap Close')) className = 'gap-group';
                else if (text.includes('21EMA')) className = 'ema-group';
                
                if (className) {
                    header.classList.add(className);
                    rows.forEach(row => {
                        const cell = row.cells[index];
                        if (cell) cell.classList.add(className);
                    });
                }
            });
        });
    });
    </script>
  </div>

  <div class="section" id="monthly-win">
    <h2>Monthly win rates</h2>
    {{ fig_monthly_5|safe }}
    {{ fig_monthly_10|safe }}
  </div>

  <div class="section" id="monthly-comparison">
    <h2>Monthly win rate (5d vs 10d)</h2>
    {{ fig_monthly_both|safe }}
  </div>

  <div class="section" id="gap-distribution">
    <h2>Gap size distribution</h2>
    {{ fig_gap_hist|safe }}
  </div>

  <div class="section" id="retracement-bucket">
    <h2>Retracement rate by gap bucket (10d)</h2>
    {{ fig_bucket|safe }}
  </div>

  <div class="section" id="win-gap">
    <h2>Win rate vs gap size</h2>
    {{ fig_gap_win_5|safe }}
    {{ fig_gap_win_10|safe }}
  </div>

  <div class="section" id="win-avwap">
    <h2>Win rate vs distance to AVWAP</h2>
    {{ fig_avwap_win_5|safe }}
    {{ fig_avwap_win_10|safe }}
  </div>

  <div class="section" id="win-21ema-above-below">
    <h2>Win rate: above vs below 21-EMA</h2>
    {{ fig_21ema_above_below|safe }}
  </div>

  <div class="section" id="win-50ema-above-below">
    <h2>Win rate: above vs below 50-EMA</h2>
    {{ fig_50ema_above_below|safe }}
  </div>

  <div class="section" id="win-100ema-above-below">
    <h2>Win rate: above vs below 100-EMA</h2>
    {{ fig_100ema_above_below|safe }}
  </div>

  <div class="section" id="win-200ema-above-below">
    <h2>Win rate: above vs below 200-EMA</h2>
    {{ fig_200ema_above_below|safe }}
  </div>

  <div class="section" id="win-avwap-current-above-below">
    <h2>Win rate: above vs below Current Quarter AVWAP</h2>
    {{ fig_avwap_current_above_below|safe }}
  </div>

  <div class="section" id="win-avwap-2q-above-below">
    <h2>Win rate: above vs below 2Q Ago AVWAP</h2>
    {{ fig_avwap_2q_above_below|safe }}
  </div>

  <div class="section" id="win-avwap-3q-above-below">
    <h2>Win rate: above vs below 3Q Ago AVWAP</h2>
    {{ fig_avwap_3q_above_below|safe }}
  </div>

  <div class="section" id="win-avwap-1y-above-below">
    <h2>Win rate: above vs below 1Y Ago AVWAP</h2>
    {{ fig_avwap_1y_above_below|safe }}
  </div>

  <div class="section" id="win-sma330-above-below">
    <h2>Win rate: above vs below 330-SMA</h2>
    {{ fig_sma330_above_below|safe }}
  </div>

  <div class="section" id="win-sma330-bucket">
    <h2>Win rate by distance to 330-SMA</h2>
    {{ fig_sma330_bucket_5|safe }}
    {{ fig_sma330_bucket_10|safe }}
  </div>

  <div class="section" id="win-spy">
    <h2>Win rate vs SPY daily change</h2>
    {{ fig_spy_bucket_5|safe }}
    {{ fig_spy_bucket_10|safe }}
  </div>

  <div class="section" id="equity-curve">
    <h2>Expected P&L (10k/trade) — Equity Curve</h2>
    <p>Total expected P&L 5d: {{ total_pnl_5d }} &nbsp;|&nbsp; 10d: {{ total_pnl_10d }}</p>
    {{ fig_equity_both|safe }}
  </div>

  <div class="section" id="gap-retrace-scatter">
    <h2>Gap size vs 10d retrace %</h2>
    {{ fig_scatter|safe }}
  </div>

  <div class="section" id="spy-retrace">
    <h2>SPY vs 10d retrace rate (by day)</h2>
    {{ fig_spy|safe }}
  </div>

  <div class="section" id="top-mfe">
    <h2>Top 20 by 10d MFE%</h2>
    {{ top_mfe|safe }}
  </div>
</body>
</html>"""


def build_report(csv_path: str, out_path: str) -> None:
    try:
        df = pd.read_csv(csv_path)
    except EmptyDataError:
        html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Gap report</title>
<style>body{{font-family:system-ui,Segoe UI,Arial;margin:16px;line-height:1.4}}</style>
</head>
<body>
  <h1>Gap report</h1>
  <p>No data found in CSV file: <code>{csv_path}</code></p>
  <p>Please run the analysis to generate a non-empty CSV and try again.</p>
</body>
</html>"""
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Wrote {out_path}")
        return

    # Infer direction from filename; fallback to unknown
    direction = 'long' if 'long' in csv_path.lower() else ('short' if 'short' in csv_path.lower() else 'unknown')

    # Parse dates and numeric columns robustly
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    pct_cols = [
        'gap_up_percent',
        'retrace_percentage_5d', 'mfe_percent_5d', 'mae_percent_5d',
        'retrace_percentage_10d', 'mfe_percent_10d', 'mae_percent_10d',
        'sma_330_distance_percent', 'spy_daily_gain_percent'
    ]
    for c in pct_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Derive booleans for retracement columns with backward compatibility
    retraced5_col = 'retraced_to_vwap_5d' if 'retraced_to_vwap_5d' in df.columns else None
    retraced10_col = 'retraced_to_vwap_10d' if 'retraced_to_vwap_10d' in df.columns else ('retraced_to_vwap' if 'retraced_to_vwap' in df.columns else None)

    # Monthly summary (changed from weekly)
    grp = None
    if 'date' in df.columns:
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly['date'].dt.to_period('M').dt.start_time
        grp = df_monthly.groupby('month')
    
    if grp is not None:
        # Use named aggregation with output column names
        agg_kwargs = {'total': ('ticker', 'count')}
        if retraced5_col:
            agg_kwargs['retraced5'] = (retraced5_col, 'sum')
        if retraced10_col:
            agg_kwargs['retraced10'] = (retraced10_col, 'sum')
        monthly = grp.agg(**agg_kwargs).reset_index()
        if 'retraced5' in monthly.columns:
            monthly['rate5'] = 100 * monthly['retraced5'] / monthly['total']
        else:
            monthly['rate5'] = np.nan
        if 'retraced10' in monthly.columns:
            monthly['rate10'] = 100 * monthly['retraced10'] / monthly['total']
        else:
            monthly['rate10'] = np.nan
        
        # Calculate 100-day (5-month) simple moving averages
        monthly['sma5'] = monthly['rate5'].rolling(window=5, min_periods=1).mean()
        monthly['sma10'] = monthly['rate10'].rolling(window=5, min_periods=1).mean()
    else:
        monthly = pd.DataFrame(columns=['month', 'total', 'retraced5', 'retraced10', 'rate5', 'rate10', 'sma5', 'sma10'])

    # Performance summary (overall and by year)
    # Define columns to use
    win_col_10 = retraced10_col
    win_col_5 = retraced5_col
    retrace_pct_10_col = 'retrace_percentage_10d' if 'retrace_percentage_10d' in df.columns else ('retrace_percentage' if 'retrace_percentage' in df.columns else None)
    retrace_pct_5_col = 'retrace_percentage_5d' if 'retrace_percentage_5d' in df.columns else None

    # Calculate gap closure metrics using the new gap_filled boolean columns
    gap_close_rate_5d = "n/a"
    gap_close_rate_10d = "n/a"
    gap_filled_5d_col = 'gap_filled_5d' if 'gap_filled_5d' in df.columns else None
    gap_filled_10d_col = 'gap_filled_10d' if 'gap_filled_10d' in df.columns else None
    
    if gap_filled_5d_col and len(df) > 0:
        gap_close_rate_5d = f"{(100.0 * pd.to_numeric(df[gap_filled_5d_col], errors='coerce').fillna(0).mean()):.2f}"
    if gap_filled_10d_col and len(df) > 0:
        gap_close_rate_10d = f"{(100.0 * pd.to_numeric(df[gap_filled_10d_col], errors='coerce').fillna(0).mean()):.2f}"

    # Calculate 21 EMA retracement rates
    ema_21_retrace_rate_5d = "n/a"
    ema_21_retrace_rate_10d = "n/a"
    if 'retraced_to_21ema_5d' in df.columns and len(df) > 0:
        ema_21_retrace_rate_5d = f"{(100.0 * pd.to_numeric(df['retraced_to_21ema_5d'], errors='coerce').fillna(0).mean()):.2f}"
    if 'retraced_to_21ema_10d' in df.columns and len(df) > 0:
        ema_21_retrace_rate_10d = f"{(100.0 * pd.to_numeric(df['retraced_to_21ema_10d'], errors='coerce').fillna(0).mean()):.2f}"

    # Build fractional retrace columns for PF computation
    if retrace_pct_5_col:
        df['retrace_frac_5d'] = pd.to_numeric(df[retrace_pct_5_col], errors='coerce') / 100.0
        df['retrace_frac_5d'] = df['retrace_frac_5d'].clip(lower=0.0, upper=1.0).fillna(0.0)
    else:
        df['retrace_frac_5d'] = np.nan

    if retrace_pct_10_col:
        df['retrace_frac_10d'] = pd.to_numeric(df[retrace_pct_10_col], errors='coerce') / 100.0
        df['retrace_frac_10d'] = df['retrace_frac_10d'].clip(lower=0.0, upper=1.0).fillna(0.0)
    else:
        df['retrace_frac_10d'] = np.nan

    # Overall win rates (5d and 10d)
    if win_col_5 and win_col_5 in df.columns and len(df) > 0:
        overall_win_rate_5d = f"{(100.0 * pd.to_numeric(df[win_col_5], errors='coerce').fillna(0).astype(float).mean()):.2f}"
    else:
        overall_win_rate_5d = "n/a"
    
    if win_col_10 and win_col_10 in df.columns and len(df) > 0:
        overall_win_rate_10d = f"{(100.0 * pd.to_numeric(df[win_col_10], errors='coerce').fillna(0).astype(float).mean()):.2f}"
    else:
        overall_win_rate_10d = "n/a"
    
    # Overall profit factors (5d and 10d)
    if retrace_pct_5_col and retrace_pct_5_col in df.columns and len(df) > 0:
        profit_units_5d = float(df['retrace_frac_5d'].sum())
        loss_units_5d = float((1.0 - df['retrace_frac_5d']).sum())
        overall_profit_factor_5d = ("∞" if profit_units_5d > 0 and loss_units_5d == 0
                                   else (f"{(profit_units_5d / loss_units_5d):.2f}" if loss_units_5d > 0 else "0.00"))
    else:
        overall_profit_factor_5d = "n/a"
    
    if retrace_pct_10_col:
        profit_units_total = float(df['retrace_frac_10d'].sum())
        loss_units_total = float((1.0 - df['retrace_frac_10d']).sum())
        overall_profit_factor_10d = ("∞" if profit_units_total > 0 and loss_units_total == 0
                                    else (f"{(profit_units_total / loss_units_total):.2f}" if loss_units_total > 0 else "0.00"))
    else:
        overall_profit_factor_10d = "n/a"

    # Yearly segmentation tables for 5d and 10d
    if 'date' in df.columns and not df['date'].isna().all():
        df_year = df.copy()
        df_year['year'] = df_year['date'].dt.year

        # 5d table
        rows5 = []
        for y, g in df_year.groupby('year', dropna=True):
            trades = int(len(g))
            
            # AVWAP retrace rate and profit factor
            if win_col_5 and win_col_5 in g.columns and trades > 0:
                avwap_rate_5 = 100.0 * pd.to_numeric(g[win_col_5], errors='coerce').fillna(0).astype(float).mean()
            else:
                avwap_rate_5 = np.nan
            if 'retrace_frac_5d' in g.columns and not g['retrace_frac_5d'].isna().all():
                p5 = float(g['retrace_frac_5d'].sum())
                l5 = float((1.0 - g['retrace_frac_5d']).sum())
                avwap_pf_5 = np.inf if p5 > 0 and l5 == 0 else (p5 / l5 if l5 > 0 else np.nan)
            else:
                avwap_pf_5 = np.nan

            # 21 EMA retrace rate and profit factor
            if 'retraced_to_21ema_5d' in g.columns and trades > 0:
                ema_rate_5 = 100.0 * pd.to_numeric(g['retraced_to_21ema_5d'], errors='coerce').fillna(0).astype(float).mean()
                # Calculate 21 EMA profit factor using binary win/loss
                ema_wins_5 = pd.to_numeric(g['retraced_to_21ema_5d'], errors='coerce').fillna(0).astype(float).sum()
                ema_losses_5 = trades - ema_wins_5
                ema_pf_5 = np.inf if ema_wins_5 > 0 and ema_losses_5 == 0 else (ema_wins_5 / ema_losses_5 if ema_losses_5 > 0 else np.nan)
            else:
                ema_rate_5 = np.nan
                ema_pf_5 = np.nan

            # Gap closure rate and profit factor using gap_filled_5d
            gap_filled_5d_col = 'gap_filled_5d' if 'gap_filled_5d' in g.columns else None
            if gap_filled_5d_col and trades > 0:
                gap_closed_5 = pd.to_numeric(g[gap_filled_5d_col], errors='coerce').fillna(0).astype(bool)
                gap_rate_5 = 100.0 * gap_closed_5.mean()
                # Calculate gap closure profit factor using binary win/loss
                gap_wins_5 = gap_closed_5.sum()
                gap_losses_5 = trades - gap_wins_5
                gap_pf_5 = np.inf if gap_wins_5 > 0 and gap_losses_5 == 0 else (gap_wins_5 / gap_losses_5 if gap_losses_5 > 0 else np.nan)
            else:
                gap_rate_5 = np.nan
                gap_pf_5 = np.nan

            rows5.append({
                'Year': int(y),
                'Trades': trades,
                'AVWAP %': round(avwap_rate_5, 2) if pd.notna(avwap_rate_5) else np.nan,
                'AVWAP PF': (round(avwap_pf_5, 2) if np.isfinite(avwap_pf_5) else (np.nan if np.isnan(avwap_pf_5) else np.inf)),
                'Gap Close %': round(gap_rate_5, 2) if pd.notna(gap_rate_5) else np.nan,
                'Gap Close PF': (round(gap_pf_5, 2) if np.isfinite(gap_pf_5) else (np.nan if np.isnan(gap_pf_5) else np.inf)),
                '21EMA %': round(ema_rate_5, 2) if pd.notna(ema_rate_5) else np.nan,
                '21EMA PF': (round(ema_pf_5, 2) if np.isfinite(ema_pf_5) else (np.nan if np.isnan(ema_pf_5) else np.inf))
            })
        perf_df_5 = pd.DataFrame(rows5).sort_values('Year') if rows5 else pd.DataFrame(columns=['Year', 'Trades', 'AVWAP %', 'AVWAP PF', 'Gap Close %', 'Gap Close PF', '21EMA %', '21EMA PF'])
        perf_table_5_html = perf_df_5.to_html(index=False, classes='perf-table', table_id='perf-5d')

        # 10d table
        rows10 = []
        for y, g in df_year.groupby('year', dropna=True):
            trades = int(len(g))
            
            # AVWAP retrace rate and profit factor
            if win_col_10 and win_col_10 in g.columns and trades > 0:
                avwap_rate_10 = 100.0 * pd.to_numeric(g[win_col_10], errors='coerce').fillna(0).astype(float).mean()
            else:
                avwap_rate_10 = np.nan
            if 'retrace_frac_10d' in g.columns and not g['retrace_frac_10d'].isna().all():
                p10 = float(g['retrace_frac_10d'].sum())
                l10 = float((1.0 - g['retrace_frac_10d']).sum())
                avwap_pf_10 = np.inf if p10 > 0 and l10 == 0 else (p10 / l10 if l10 > 0 else np.nan)
            else:
                avwap_pf_10 = np.nan

            # 21 EMA retrace rate and profit factor
            if 'retraced_to_21ema_10d' in g.columns and trades > 0:
                ema_rate_10 = 100.0 * pd.to_numeric(g['retraced_to_21ema_10d'], errors='coerce').fillna(0).astype(float).mean()
                # Calculate 21 EMA profit factor using binary win/loss
                ema_wins_10 = pd.to_numeric(g['retraced_to_21ema_10d'], errors='coerce').fillna(0).astype(float).sum()
                ema_losses_10 = trades - ema_wins_10
                ema_pf_10 = np.inf if ema_wins_10 > 0 and ema_losses_10 == 0 else (ema_wins_10 / ema_losses_10 if ema_losses_10 > 0 else np.nan)
            else:
                ema_rate_10 = np.nan
                ema_pf_10 = np.nan

            # Gap closure rate and profit factor using gap_filled_10d
            gap_filled_10d_col = 'gap_filled_10d' if 'gap_filled_10d' in g.columns else None
            if gap_filled_10d_col and trades > 0:
                gap_closed_10 = pd.to_numeric(g[gap_filled_10d_col], errors='coerce').fillna(0).astype(bool)
                gap_rate_10 = 100.0 * gap_closed_10.mean()
                # Calculate gap closure profit factor using binary win/loss
                gap_wins_10 = gap_closed_10.sum()
                gap_losses_10 = trades - gap_wins_10
                gap_pf_10 = np.inf if gap_wins_10 > 0 and gap_losses_10 == 0 else (gap_wins_10 / gap_losses_10 if gap_losses_10 > 0 else np.nan)
            else:
                gap_rate_10 = np.nan
                gap_pf_10 = np.nan

            rows10.append({
                'Year': int(y),
                'Trades': trades,
                'AVWAP %': round(avwap_rate_10, 2) if pd.notna(avwap_rate_10) else np.nan,
                'AVWAP PF': (round(avwap_pf_10, 2) if np.isfinite(avwap_pf_10) else (np.nan if np.isnan(avwap_pf_10) else np.inf)),
                'Gap Close %': round(gap_rate_10, 2) if pd.notna(gap_rate_10) else np.nan,
                'Gap Close PF': (round(gap_pf_10, 2) if np.isfinite(gap_pf_10) else (np.nan if np.isnan(gap_pf_10) else np.inf)),
                '21EMA %': round(ema_rate_10, 2) if pd.notna(ema_rate_10) else np.nan,
                '21EMA PF': (round(ema_pf_10, 2) if np.isfinite(ema_pf_10) else (np.nan if np.isnan(ema_pf_10) else np.inf))
            })
        perf_df_10 = pd.DataFrame(rows10).sort_values('Year') if rows10 else pd.DataFrame(columns=['Year', 'Trades', 'AVWAP %', 'AVWAP PF', 'Gap Close %', 'Gap Close PF', '21EMA %', '21EMA PF'])
        perf_table_10_html = perf_df_10.to_html(index=False, classes='perf-table', table_id='perf-10d')
    else:
        perf_table_5_html = pd.DataFrame(columns=['Year', 'Trades', 'AVWAP %', 'AVWAP PF', 'Gap Close %', 'Gap Close PF', '21EMA %', '21EMA PF']).to_html(index=False, classes='perf-table')
        perf_table_10_html = pd.DataFrame(columns=['Year', 'Trades', 'AVWAP %', 'AVWAP PF', 'Gap Close %', 'Gap Close PF', '21EMA %', '21EMA PF']).to_html(index=False, classes='perf-table')

    # Charts (changed to monthly with SMA overlay)
    if not monthly.empty:
        import plotly.graph_objects as go
        
        # 5d monthly chart with SMA
        fig_daily_5 = go.Figure()
        fig_daily_5.add_trace(go.Scatter(x=monthly['month'], y=monthly['rate5'], mode='lines', 
                                        name='Monthly 5d rate', line=dict(color='blue')))
        fig_daily_5.add_trace(go.Scatter(x=monthly['month'], y=monthly['sma5'], mode='lines', 
                                        name='100-day SMA', line=dict(color='red', width=2)))
        fig_daily_5.update_layout(title='Monthly 5d retracement rate (%) with 100-day SMA', 
                                 xaxis_title='Month', yaxis_title='Rate (%)')
        
        # 10d monthly chart with SMA
        fig_daily_10 = go.Figure()
        fig_daily_10.add_trace(go.Scatter(x=monthly['month'], y=monthly['rate10'], mode='lines', 
                                         name='Monthly 10d rate', line=dict(color='blue')))
        fig_daily_10.add_trace(go.Scatter(x=monthly['month'], y=monthly['sma10'], mode='lines', 
                                         name='100-day SMA', line=dict(color='red', width=2)))
        fig_daily_10.update_layout(title='Monthly 10d retracement rate (%) with 100-day SMA', 
                                  xaxis_title='Month', yaxis_title='Rate (%)')
    else:
        fig_daily_5 = px.scatter(title='No monthly data')
        fig_daily_10 = px.scatter(title='No monthly data')

    # Monthly win rate charts (5d and 10d) - changed to line charts with 14-period MA
    if 'date' in df.columns and not df['date'].isna().all():
        df_m = df.copy()
        df_m['month'] = df_m['date'].dt.to_period('M').dt.to_timestamp()
        monthly_5 = None
        monthly_10 = None
        
        import plotly.graph_objects as go
        
        if retraced5_col:
            df_m['retr5_num'] = pd.to_numeric(df_m[retraced5_col], errors='coerce').fillna(0).astype(float)
            monthly_5 = (df_m.groupby('month', dropna=True)['retr5_num'].mean().reset_index(name='rate'))
            monthly_5['rate'] = 100 * monthly_5['rate']
            monthly_5['ma14'] = monthly_5['rate'].rolling(window=14, min_periods=1).mean()
            
            fig_monthly_5 = go.Figure()
            fig_monthly_5.add_trace(go.Scatter(x=monthly_5['month'], y=monthly_5['rate'], mode='lines', 
                                              name='Monthly 5d win rate', line=dict(color='blue')))
            fig_monthly_5.add_trace(go.Scatter(x=monthly_5['month'], y=monthly_5['ma14'], mode='lines', 
                                              name='14-period MA', line=dict(color='red', width=2)))
            fig_monthly_5.update_layout(title='Monthly 5d win rate (%) with 14-period MA', 
                                       xaxis_title='Month', yaxis_title='Rate (%)')
        else:
            fig_monthly_5 = px.scatter(title='Monthly 5d win rate (n/a)')
            
        if retraced10_col:
            df_m['retr10_num'] = pd.to_numeric(df_m[retraced10_col], errors='coerce').fillna(0).astype(float)
            monthly_10 = (df_m.groupby('month', dropna=True)['retr10_num'].mean().reset_index(name='rate'))
            monthly_10['rate'] = 100 * monthly_10['rate']
            monthly_10['ma14'] = monthly_10['rate'].rolling(window=14, min_periods=1).mean()
            
            fig_monthly_10 = go.Figure()
            fig_monthly_10.add_trace(go.Scatter(x=monthly_10['month'], y=monthly_10['rate'], mode='lines', 
                                               name='Monthly 10d win rate', line=dict(color='blue')))
            fig_monthly_10.add_trace(go.Scatter(x=monthly_10['month'], y=monthly_10['ma14'], mode='lines', 
                                               name='14-period MA', line=dict(color='red', width=2)))
            fig_monthly_10.update_layout(title='Monthly 10d win rate (%) with 14-period MA', 
                                        xaxis_title='Month', yaxis_title='Rate (%)')
        else:
            fig_monthly_10 = px.scatter(title='Monthly 10d win rate (n/a)')

        # Combined monthly win rate line chart (5d vs 10d)
        if (isinstance(monthly_5, pd.DataFrame) and not monthly_5.empty) or (isinstance(monthly_10, pd.DataFrame) and not monthly_10.empty):
            parts = []
            if isinstance(monthly_5, pd.DataFrame) and not monthly_5.empty:
                parts.append(monthly_5.assign(horizon='5d'))
            if isinstance(monthly_10, pd.DataFrame) and not monthly_10.empty:
                parts.append(monthly_10.assign(horizon='10d'))
            monthly_both = pd.concat(parts, ignore_index=True)
            fig_monthly_both = px.line(monthly_both, x='month', y='rate', color='horizon', markers=True, title='Monthly win rate (5d vs 10d, %)')
        else:
            fig_monthly_both = px.scatter(title='Monthly win rate (5d vs 10d) (n/a)')
    else:
        import plotly.graph_objects as go
        fig_monthly_5 = go.Figure().update_layout(title='Monthly 5d win rate (no dates)')
        fig_monthly_10 = go.Figure().update_layout(title='Monthly 10d win rate (no dates)')
        fig_monthly_both = go.Figure().update_layout(title='Monthly win rate (5d vs 10d) (no dates)')

    # Gap histogram
    if 'gap_up_percent' in df.columns and not df.empty:
        fig_gap_hist = px.histogram(df, x='gap_up_percent', nbins=40, title='Gap size (%)', marginal='box')
    else:
        fig_gap_hist = px.scatter(title='No gap data')

    # Buckets and rates (10d)
    if 'gap_up_percent' in df.columns and retraced10_col:
        bins = [5, 7.5, 10, 15, 25, 50, 100]
        labels = ['5–7.5', '7.5–10', '10–15', '15–25', '25–50', '50–100']
        dfb = df.copy()
        dfb['gap_bucket'] = pd.cut(dfb['gap_up_percent'], bins=bins, labels=labels, right=False)
        bucket = (dfb.dropna(subset=['gap_bucket'])
                    .groupby('gap_bucket', observed=False)
                    .agg(total=('ticker', 'count'),
                         retraced10=(retraced10_col, 'mean'))
                    .reset_index())
        bucket['rate10'] = 100 * bucket['retraced10']
        import plotly.graph_objects as go
        fig_bucket = go.Figure(data=[
            go.Bar(x=bucket['gap_bucket'], y=bucket['rate10'],
                   text=[f'{rate:.1f}%' for rate in bucket['rate10']],
                   textposition='outside')
        ])
        fig_bucket.update_layout(title='10d retracement rate by gap bucket (%)')
    else:
        fig_bucket = px.scatter(title='No bucket data')

    # Win rate vs gap size (5d and 10d)
    if 'gap_up_percent' in df.columns and (retraced5_col or retraced10_col):
        dfw = df.copy()
        # Prepare win columns
        if retraced5_col:
            dfw['win5'] = pd.to_numeric(dfw[retraced5_col], errors='coerce').fillna(0).astype(float)
        if retraced10_col:
            dfw['win10'] = pd.to_numeric(dfw[retraced10_col], errors='coerce').fillna(0).astype(float)
        dfw['gap_bucket'] = pd.cut(dfw['gap_up_percent'], bins=bins, labels=labels, right=False)
        gb = dfw.dropna(subset=['gap_bucket']).groupby('gap_bucket', observed=False)
        if retraced5_col:
            gap_win_5 = (gb['win5'].mean().reset_index(name='rate'))
            gap_win_5['rate'] = 100 * gap_win_5['rate']
            import plotly.graph_objects as go
            fig_gap_win_5 = go.Figure(data=[
                go.Bar(x=gap_win_5['gap_bucket'], y=gap_win_5['rate'],
                       text=[f'{rate:.1f}%' for rate in gap_win_5['rate']],
                       textposition='outside')
            ])
            fig_gap_win_5.update_layout(title='Win rate vs gap size (5d, %)')
        else:
            fig_gap_win_5 = px.scatter(title='Win rate vs gap size (5d unavailable)')
        if retraced10_col:
            gap_win_10 = (gb['win10'].mean().reset_index(name='rate'))
            gap_win_10['rate'] = 100 * gap_win_10['rate']
            import plotly.graph_objects as go
            fig_gap_win_10 = go.Figure(data=[
                go.Bar(x=gap_win_10['gap_bucket'], y=gap_win_10['rate'],
                       text=[f'{rate:.1f}%' for rate in gap_win_10['rate']],
                       textposition='outside')
            ])
            fig_gap_win_10.update_layout(title='Win rate vs gap size (10d, %)')
        else:
            fig_gap_win_10 = px.scatter(title='Win rate vs gap size (10d unavailable)')
    else:
        import plotly.graph_objects as go
        fig_gap_win_5 = go.Figure().update_layout(title='Win rate vs gap size (5d) — no data')
        fig_gap_win_10 = go.Figure().update_layout(title='Win rate vs gap size (10d) — no data')

    # Win rate vs distance to AVWAP (5d and 10d)
    if all(col in df.columns for col in ['anchored_vwap', 'open_price']) and (retraced5_col or retraced10_col):
        dfv = df.copy()
        # Use per-row direction if available, else infer from filename direction variable
        if 'direction' in dfv.columns:
            dir_series = dfv['direction'].astype(str).str.lower()
            dist = np.where(dir_series == 'short', dfv['open_price'] - dfv['anchored_vwap'],
                            dfv['anchored_vwap'] - dfv['open_price'])
        else:
            if direction == 'short':
                dist = dfv['open_price'] - dfv['anchored_vwap']
            else:
                dist = dfv['anchored_vwap'] - dfv['open_price']
        dfv['dist_pct'] = 100.0 * (pd.to_numeric(dist, errors='coerce')) / pd.to_numeric(dfv['open_price'], errors='coerce')
        dfv['dist_pct'] = dfv['dist_pct'].clip(lower=0)
        dist_bins = [0, 1, 2, 3, 5, 8, 12, 20, 1000]
        dist_labels = ['0–1', '1–2', '2–3', '3–5', '5–8', '8–12', '12–20', '20%+']
        dfv['dist_bucket'] = pd.cut(dfv['dist_pct'], bins=dist_bins, labels=dist_labels, right=False)
        if retraced5_col:
            dfv['win5'] = pd.to_numeric(dfv[retraced5_col], errors='coerce').fillna(0).astype(float)
        if retraced10_col:
            dfv['win10'] = pd.to_numeric(dfv[retraced10_col], errors='coerce').fillna(0).astype(float)
        gv = dfv.dropna(subset=['dist_bucket']).groupby('dist_bucket', observed=False)
        if retraced5_col:
            avwap_win_5 = (gv['win5'].mean().reset_index(name='rate'))
            avwap_win_5['rate'] = 100 * avwap_win_5['rate']
            import plotly.graph_objects as go
            fig_avwap_win_5 = go.Figure(data=[
                go.Bar(x=avwap_win_5['dist_bucket'], y=avwap_win_5['rate'],
                       text=[f'{rate:.1f}%' for rate in avwap_win_5['rate']],
                       textposition='outside')
            ])
            fig_avwap_win_5.update_layout(title='Win rate vs distance to AVWAP (5d, %)')
        else:
            fig_avwap_win_5 = px.scatter(title='Win rate vs distance to AVWAP (5d unavailable)')
        if retraced10_col:
            avwap_win_10 = (gv['win10'].mean().reset_index(name='rate'))
            avwap_win_10['rate'] = 100 * avwap_win_10['rate']
            import plotly.graph_objects as go
            fig_avwap_win_10 = go.Figure(data=[
                go.Bar(x=avwap_win_10['dist_bucket'], y=avwap_win_10['rate'],
                       text=[f'{rate:.1f}%' for rate in avwap_win_10['rate']],
                       textposition='outside')
            ])
            fig_avwap_win_10.update_layout(title='Win rate vs distance to AVWAP (10d, %)')
        else:
            fig_avwap_win_10 = px.scatter(title='Win rate vs distance to AVWAP (10d unavailable)')
    else:
        import plotly.graph_objects as go
        fig_avwap_win_5 = go.Figure().update_layout(title='Win rate vs distance to AVWAP (5d) — no data')
        fig_avwap_win_10 = go.Figure().update_layout(title='Win rate vs distance to AVWAP (10d) — no data')

    # Helper function for EMA/SMA above/below analysis
    def create_above_below_chart(df, distance_col, label, retraced5_col, retraced10_col):
        if distance_col in df.columns and (retraced5_col or retraced10_col):
            dfx = df.copy()
            dfx['dist'] = pd.to_numeric(dfx[distance_col], errors='coerce')
            dfx = dfx.dropna(subset=['dist'])
            if retraced5_col:
                dfx['win5'] = pd.to_numeric(dfx[retraced5_col], errors='coerce').fillna(0).astype(float)
            if retraced10_col:
                dfx['win10'] = pd.to_numeric(dfx[retraced10_col], errors='coerce').fillna(0).astype(float)

            # Above vs Below
            dfx['above_below'] = np.where(dfx['dist'] >= 0, f'Above {label}', f'Below {label}')
            parts = []
            gb_ab = dfx.groupby('above_below', observed=False)
            if retraced5_col:
                r5 = gb_ab['win5'].mean().reset_index(name='rate')
                r5['horizon'] = '5d'
                parts.append(r5)
            if retraced10_col:
                r10 = gb_ab['win10'].mean().reset_index(name='rate')
                r10['horizon'] = '10d'
                parts.append(r10)
            if parts:
                ab_long = pd.concat(parts, ignore_index=True)
                ab_long['rate'] = 100 * ab_long['rate']
                import plotly.graph_objects as go
                fig = go.Figure()
                
                # Group data by horizon for grouped bars
                for horizon in ab_long['horizon'].unique():
                    horizon_data = ab_long[ab_long['horizon'] == horizon]
                    fig.add_trace(go.Bar(
                        x=horizon_data['above_below'],
                        y=horizon_data['rate'],
                        name=horizon,
                        text=[f'{rate:.1f}%' for rate in horizon_data['rate']],
                        textposition='outside'
                    ))
                
                fig.update_layout(
                    title=f'Win rate: above vs below {label} (%)',
                    barmode='group'
                )
                return fig
            else:
                return px.scatter(title=f'Win rate: above vs below {label} (n/a)')
        else:
            return px.scatter(title=f'Win rate: above vs below {label} (no data)')

    # Win rate vs various EMAs/SMAs (above/below)
    fig_21ema_above_below = create_above_below_chart(df, 'sma_21_distance_percent', '21-EMA', retraced5_col, retraced10_col)
    fig_50ema_above_below = create_above_below_chart(df, 'sma_50_distance_percent', '50-EMA', retraced5_col, retraced10_col)
    fig_100ema_above_below = create_above_below_chart(df, 'sma_100_distance_percent', '100-EMA', retraced5_col, retraced10_col)
    fig_200ema_above_below = create_above_below_chart(df, 'sma_200_distance_percent', '200-EMA', retraced5_col, retraced10_col)

    # Win rate vs quarterly AVWAPs (above/below)
    fig_avwap_current_above_below = create_above_below_chart(df, 'avwap_current_distance_percent', 'Current Quarter AVWAP', retraced5_col, retraced10_col)
    fig_avwap_2q_above_below = create_above_below_chart(df, 'avwap_2q_ago_distance_percent', '2Q Ago AVWAP', retraced5_col, retraced10_col)
    fig_avwap_3q_above_below = create_above_below_chart(df, 'avwap_3q_ago_distance_percent', '3Q Ago AVWAP', retraced5_col, retraced10_col)
    fig_avwap_1y_above_below = create_above_below_chart(df, 'avwap_1y_ago_distance_percent', '1Y Ago AVWAP', retraced5_col, retraced10_col)

    # Win rate vs 330-SMA (above/below) and by distance buckets
    if 'sma_330_distance_percent' in df.columns and (retraced5_col or retraced10_col):
        dfx = df.copy()
        dfx['sma_dist'] = pd.to_numeric(dfx['sma_330_distance_percent'], errors='coerce')
        dfx = dfx.dropna(subset=['sma_dist'])
        if retraced5_col:
            dfx['win5'] = pd.to_numeric(dfx[retraced5_col], errors='coerce').fillna(0).astype(float)
        if retraced10_col:
            dfx['win10'] = pd.to_numeric(dfx[retraced10_col], errors='coerce').fillna(0).astype(float)

        # Above vs Below 330-SMA
        dfx['above_below'] = np.where(dfx['sma_dist'] >= 0, 'Above 330-SMA', 'Below 330-SMA')
        parts = []
        gb_ab = dfx.groupby('above_below', observed=False)
        if retraced5_col:
            r5 = gb_ab['win5'].mean().reset_index(name='rate')
            r5['horizon'] = '5d'
            parts.append(r5)
        if retraced10_col:
            r10 = gb_ab['win10'].mean().reset_index(name='rate')
            r10['horizon'] = '10d'
            parts.append(r10)
        if parts:
            ab_long = pd.concat(parts, ignore_index=True)
            ab_long['rate'] = 100 * ab_long['rate']
            import plotly.graph_objects as go
            fig_sma330_above_below = go.Figure()
            
            # Group data by horizon for grouped bars
            for horizon in ab_long['horizon'].unique():
                horizon_data = ab_long[ab_long['horizon'] == horizon]
                fig_sma330_above_below.add_trace(go.Bar(
                    x=horizon_data['above_below'],
                    y=horizon_data['rate'],
                    name=horizon,
                    text=[f'{rate:.1f}%' for rate in horizon_data['rate']],
                    textposition='outside'
                ))
            
            fig_sma330_above_below.update_layout(
                title='Win rate: above vs below 330-SMA (%)',
                barmode='group'
            )
        else:
            fig_sma330_above_below = px.scatter(title='Win rate: above vs below 330-SMA (n/a)')

        # Buckets by signed distance to 330-SMA
        bins = [-1000, -20, -12, -8, -5, -3, -1, 0, 1, 3, 5, 8, 12, 20, 1000]
        labels = ['<=-20', '-20–-12', '-12–-8', '-8–-5', '-5–-3', '-3–-1', '-1–0',
                  '0–1', '1–3', '3–5', '5–8', '8–12', '12–20', '>=20']
        dfx['sma_bucket'] = pd.cut(dfx['sma_dist'], bins=bins, labels=labels, right=False, include_lowest=True)
        gv = dfx.dropna(subset=['sma_bucket']).groupby('sma_bucket', observed=False)
        if retraced5_col:
            b5 = gv['win5'].mean().reset_index(name='rate')
            b5['rate'] = 100 * b5['rate']
            import plotly.graph_objects as go
            fig_sma330_bucket_5 = go.Figure(data=[
                go.Bar(x=b5['sma_bucket'], y=b5['rate'],
                       text=[f'{rate:.1f}%' for rate in b5['rate']],
                       textposition='outside')
            ])
            fig_sma330_bucket_5.update_layout(title='Win rate vs distance to 330-SMA (5d, %)')
        else:
            fig_sma330_bucket_5 = px.scatter(title='Win rate vs distance to 330-SMA (5d unavailable)')
        if retraced10_col:
            b10 = gv['win10'].mean().reset_index(name='rate')
            b10['rate'] = 100 * b10['rate']
            import plotly.graph_objects as go
            fig_sma330_bucket_10 = go.Figure(data=[
                go.Bar(x=b10['sma_bucket'], y=b10['rate'],
                       text=[f'{rate:.1f}%' for rate in b10['rate']],
                       textposition='outside')
            ])
            fig_sma330_bucket_10.update_layout(title='Win rate vs distance to 330-SMA (10d, %)')
        else:
            fig_sma330_bucket_10 = px.scatter(title='Win rate vs distance to 330-SMA (10d unavailable)')
    else:
        import plotly.graph_objects as go
        fig_sma330_above_below = px.scatter(title='Win rate: above vs below 330-SMA (no data)')
        fig_sma330_bucket_5 = px.scatter(title='Win rate vs distance to 330-SMA (5d) — no data')
        fig_sma330_bucket_10 = px.scatter(title='Win rate vs distance to 330-SMA (10d) — no data')

    # Win rate vs SPY daily change (5d and 10d)
    if 'spy_daily_gain_percent' in df.columns and (retraced5_col or retraced10_col):
        dfs = df.copy()
        dfs['spy_pct'] = pd.to_numeric(dfs['spy_daily_gain_percent'], errors='coerce')
        spy_bins = [-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5]
        spy_labels = ['<-3', '-3–-2', '-2–-1', '-1–-0.5', '-0.5–0', '0–0.5', '0.5–1', '1–2', '2–3', '>3']
        dfs['spy_bucket'] = pd.cut(dfs['spy_pct'], bins=spy_bins, labels=spy_labels, right=False)
        if retraced5_col:
            dfs['win5'] = pd.to_numeric(dfs[retraced5_col], errors='coerce').fillna(0).astype(float)
        if retraced10_col:
            dfs['win10'] = pd.to_numeric(dfs[retraced10_col], errors='coerce').fillna(0).astype(float)
        gs = dfs.dropna(subset=['spy_bucket']).groupby('spy_bucket', observed=False)
        if retraced5_col:
            spy_win_5 = gs['win5'].mean().reset_index(name='rate')
            spy_win_5['rate'] = 100 * spy_win_5['rate']
            import plotly.graph_objects as go
            fig_spy_bucket_5 = go.Figure(data=[
                go.Bar(x=spy_win_5['spy_bucket'], y=spy_win_5['rate'],
                       text=[f'{rate:.1f}%' for rate in spy_win_5['rate']],
                       textposition='outside')
            ])
            fig_spy_bucket_5.update_layout(title='Win rate vs SPY daily change (5d, %)')
        else:
            fig_spy_bucket_5 = px.scatter(title='Win rate vs SPY daily change (5d unavailable)')
        if retraced10_col:
            spy_win_10 = gs['win10'].mean().reset_index(name='rate')
            spy_win_10['rate'] = 100 * spy_win_10['rate']
            import plotly.graph_objects as go
            fig_spy_bucket_10 = go.Figure(data=[
                go.Bar(x=spy_win_10['spy_bucket'], y=spy_win_10['rate'],
                       text=[f'{rate:.1f}%' for rate in spy_win_10['rate']],
                       textposition='outside')
            ])
            fig_spy_bucket_10.update_layout(title='Win rate vs SPY daily change (10d, %)')
        else:
            fig_spy_bucket_10 = px.scatter(title='Win rate vs SPY daily change (10d unavailable)')
    else:
        import plotly.graph_objects as go
        fig_spy_bucket_5 = px.scatter(title='Win rate vs SPY daily change (5d) — no data')
        fig_spy_bucket_10 = px.scatter(title='Win rate vs SPY daily change (10d) — no data')

    # Expected P&L (10k per trade) and equity curve (5d vs 10d)
    # Assumptions:
    # - Entry at open_price on gap day
    # - Exit when/if AVWAP is reached within horizon, otherwise best favorable extreme within horizon
    # - Expected P&L uses symmetric model: shares * distance * (2 * retrace_fraction - 1), capped to [-600, 1200] per trade
    if all(col in df.columns for col in ['anchored_vwap', 'open_price']) and (('retrace_percentage_5d' in df.columns) or ('retrace_percentage_10d' in df.columns or 'retrace_percentage' in df.columns)):
        dfp = df.copy()
        dfp['open_price'] = pd.to_numeric(dfp['open_price'], errors='coerce')
        dfp['anchored_vwap'] = pd.to_numeric(dfp['anchored_vwap'], errors='coerce')
        dfp['distance'] = (dfp['anchored_vwap'] - dfp['open_price']).abs()
        dfp['shares'] = np.where(dfp['open_price'] > 0, 10000.0 / dfp['open_price'], 0.0)

        if 'retrace_percentage_5d' in dfp.columns:
            f5 = pd.to_numeric(dfp['retrace_percentage_5d'], errors='coerce') / 100.0
            raw_pnl_5 = dfp['shares'] * dfp['distance'] * (2.0 * f5.clip(lower=0.0, upper=1.0) - 1.0)
            dfp['pnl_5'] = raw_pnl_5.clip(lower=-600.0, upper=1200.0)
        else:
            dfp['pnl_5'] = np.nan

        r10_col = 'retrace_percentage_10d' if 'retrace_percentage_10d' in dfp.columns else ('retrace_percentage' if 'retrace_percentage' in dfp.columns else None)
        if r10_col:
            f10 = pd.to_numeric(dfp[r10_col], errors='coerce') / 100.0
            raw_pnl_10 = dfp['shares'] * dfp['distance'] * (2.0 * f10.clip(lower=0.0, upper=1.0) - 1.0)
            dfp['pnl_10'] = raw_pnl_10.clip(lower=-600.0, upper=1200.0)
        else:
            dfp['pnl_10'] = np.nan

        # Build equity curves by cumulative sum ordered by date (or original order if date missing)
        if 'date' in dfp.columns and not dfp['date'].isna().all():
            dfp_sorted = dfp.sort_values('date')
            x_axis = pd.to_datetime(dfp_sorted['date'], errors='coerce')
        else:
            dfp_sorted = dfp.copy()
            dfp_sorted['_idx'] = np.arange(len(dfp_sorted))
            x_axis = dfp_sorted['_idx']

        equity_parts = []
        if 'pnl_5' in dfp_sorted.columns and not dfp_sorted['pnl_5'].isna().all():
            equity_5 = pd.DataFrame({'x': x_axis, 'equity': dfp_sorted['pnl_5'].fillna(0).cumsum(), 'horizon': '5d'})
            equity_parts.append(equity_5)
        if 'pnl_10' in dfp_sorted.columns and not dfp_sorted['pnl_10'].isna().all():
            equity_10 = pd.DataFrame({'x': x_axis, 'equity': dfp_sorted['pnl_10'].fillna(0).cumsum(), 'horizon': '10d'})
            equity_parts.append(equity_10)

        if equity_parts:
            equity_df = pd.concat(equity_parts, ignore_index=True)
            fig_equity_both = px.line(equity_df, x='x', y='equity', color='horizon', markers=False, title='Equity Curve (Expected P&L, $10k per trade)')
        else:
            fig_equity_both = px.scatter(title='Equity Curve (no data)')

        total_pnl_5d = (f"${dfp['pnl_5'].sum():,.0f}" if 'pnl_5' in dfp.columns and not dfp['pnl_5'].isna().all() else "n/a")
        total_pnl_10d = (f"${dfp['pnl_10'].sum():,.0f}" if 'pnl_10' in dfp.columns and not dfp['pnl_10'].isna().all() else "n/a")
    else:
        import plotly.graph_objects as go
        fig_equity_both = go.Figure().update_layout(title='Equity Curve (insufficient data)')
        total_pnl_5d = "n/a"
        total_pnl_10d = "n/a"

    # Gap vs retrace scatter (10d)
    ycol = 'retrace_percentage_10d' if 'retrace_percentage_10d' in df.columns else None
    color_col = retraced10_col
    if ycol and color_col:
        fig_scatter = px.scatter(df, x='gap_up_percent', y=ycol, color=color_col,
                                 title='Gap size vs 10d retrace %',
                                 hover_data=[c for c in ['ticker', 'date'] if c in df.columns])
    else:
        fig_scatter = px.scatter(title='No retrace percentage data')

    # SPY vs 10d retrace rate (by day)
    if 'spy_daily_gain_percent' in df.columns and retraced10_col and not df.empty:
        spy = (df.groupby('date')
                 .agg(spy=('spy_daily_gain_percent', 'mean'),
                      rate10=(retraced10_col, 'mean'))
                 .reset_index())
        spy['rate10'] = spy['rate10'] * 100  # mean of booleans -> percentage
        fig_spy = px.scatter(spy, x='spy', y='rate10', trendline='ols',
                             title='SPY daily gain vs 10d retrace rate (%)')
    else:
        import plotly.graph_objects as go
        fig_spy = go.Figure().update_layout(title='SPY not available (skipped)')

    # Top 20 by 10d MFE%
    sort_col = 'mfe_percent_10d' if 'mfe_percent_10d' in df.columns else ('mfe_percent' if 'mfe_percent' in df.columns else None)
    top_cols = [c for c in ['ticker', 'date', 'gap_up_percent', 'mfe_percent_10d', 'mae_percent_10d', 'retrace_percentage_10d'] if c in df.columns]
    if sort_col and not df.empty:
        present_cols = [c for c in top_cols if c in df.columns]
        df_sorted = df.sort_values(by=sort_col, ascending=False)
        if present_cols:
            top = df_sorted[present_cols].head(20)
        else:
            fallback_cols = [c for c in ['ticker', 'date', 'gap_up_percent'] if c in df_sorted.columns]
            top = df_sorted[fallback_cols].head(20) if fallback_cols else df_sorted.head(20)
        top_html = top.to_html(index=False)
    else:
        top_html = pd.DataFrame(columns=['No data']).to_html(index=False)

    html = Template(TEMPLATE).render(
        title=f"Gap {direction} report",
        nrows=len(df),
        ndates=df['date'].nunique() if 'date' in df.columns else 'n/a',
        direction=direction,
        overall_win_rate_5d=overall_win_rate_5d,
        overall_win_rate_10d=overall_win_rate_10d,
        overall_profit_factor_5d=overall_profit_factor_5d,
        overall_profit_factor_10d=overall_profit_factor_10d,
        gap_close_rate_5d=gap_close_rate_5d,
        gap_close_rate_10d=gap_close_rate_10d,
        ema_21_retrace_rate_5d=ema_21_retrace_rate_5d,
        ema_21_retrace_rate_10d=ema_21_retrace_rate_10d,
        perf_table_5=perf_table_5_html,
        perf_table_10=perf_table_10_html,
        fig_monthly_5=fig_monthly_5.to_html(include_plotlyjs='cdn', full_html=False),
        fig_monthly_10=fig_monthly_10.to_html(include_plotlyjs=False, full_html=False),
        fig_monthly_both=fig_monthly_both.to_html(include_plotlyjs=False, full_html=False),
        fig_gap_hist=fig_gap_hist.to_html(include_plotlyjs=False, full_html=False),
        fig_bucket=fig_bucket.to_html(include_plotlyjs=False, full_html=False),
        fig_gap_win_5=fig_gap_win_5.to_html(include_plotlyjs=False, full_html=False),
        fig_gap_win_10=fig_gap_win_10.to_html(include_plotlyjs=False, full_html=False),
        fig_avwap_win_5=fig_avwap_win_5.to_html(include_plotlyjs=False, full_html=False),
        fig_avwap_win_10=fig_avwap_win_10.to_html(include_plotlyjs=False, full_html=False),
        fig_21ema_above_below=fig_21ema_above_below.to_html(include_plotlyjs=False, full_html=False),
        fig_50ema_above_below=fig_50ema_above_below.to_html(include_plotlyjs=False, full_html=False),
        fig_100ema_above_below=fig_100ema_above_below.to_html(include_plotlyjs=False, full_html=False),
        fig_200ema_above_below=fig_200ema_above_below.to_html(include_plotlyjs=False, full_html=False),
        fig_avwap_current_above_below=fig_avwap_current_above_below.to_html(include_plotlyjs=False, full_html=False),
        fig_avwap_2q_above_below=fig_avwap_2q_above_below.to_html(include_plotlyjs=False, full_html=False),
        fig_avwap_3q_above_below=fig_avwap_3q_above_below.to_html(include_plotlyjs=False, full_html=False),
        fig_avwap_1y_above_below=fig_avwap_1y_above_below.to_html(include_plotlyjs=False, full_html=False),
        fig_sma330_above_below=fig_sma330_above_below.to_html(include_plotlyjs=False, full_html=False),
        fig_sma330_bucket_5=fig_sma330_bucket_5.to_html(include_plotlyjs=False, full_html=False),
        fig_sma330_bucket_10=fig_sma330_bucket_10.to_html(include_plotlyjs=False, full_html=False),
        fig_spy_bucket_5=fig_spy_bucket_5.to_html(include_plotlyjs=False, full_html=False),
        fig_spy_bucket_10=fig_spy_bucket_10.to_html(include_plotlyjs=False, full_html=False),
        fig_equity_both=fig_equity_both.to_html(include_plotlyjs=False, full_html=False),
        total_pnl_5d=total_pnl_5d,
        total_pnl_10d=total_pnl_10d,
        fig_scatter=fig_scatter.to_html(include_plotlyjs=False, full_html=False),
        fig_spy=fig_spy.to_html(include_plotlyjs=False, full_html=False),
        top_mfe=top_html
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description='Generate an HTML report from gap analysis CSV output')
    ap.add_argument('--csv', required=True, help='Path to output-short.csv or output-long.csv')
    ap.add_argument('--out', default=None, help='Output HTML path (default: report-<direction>.html)')
    args = ap.parse_args()
    direction = 'long' if 'long' in args.csv.lower() else ('short' if 'short' in args.csv.lower() else 'report')
    out = args.out or f"report-{direction}.html"
    build_report(args.csv, out)


if __name__ == '__main__':
    main()
