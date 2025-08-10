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
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <p class="meta">Rows: {{ nrows }}, Dates: {{ ndates }}, Direction: {{ direction }}</p>

  <div class="section">
    <h2>Performance summary</h2>
    <p>Overall win rate: {{ overall_win_rate }}% &nbsp;|&nbsp; Profit factor: {{ overall_profit_factor }}</p>
    <h3>5-day performance by year</h3>
    {{ perf_table_5|safe }}
    <h3>10-day performance by year</h3>
    {{ perf_table_10|safe }}
  </div>

  <div class="section">
    <h2>Daily retracement rates</h2>
    {{ fig_daily_5|safe }}
    {{ fig_daily_10|safe }}
  </div>

  <div class="section">
    <h2>Monthly win rates</h2>
    {{ fig_monthly_5|safe }}
    {{ fig_monthly_10|safe }}
  </div>

  <div class="section">
    <h2>Monthly win rate (5d vs 10d)</h2>
    {{ fig_monthly_both|safe }}
  </div>

  <div class="section">
    <h2>Gap size distribution</h2>
    {{ fig_gap_hist|safe }}
  </div>

  <div class="section">
    <h2>Retracement rate by gap bucket (10d)</h2>
    {{ fig_bucket|safe }}
  </div>

  <div class="section">
    <h2>Win rate vs gap size</h2>
    {{ fig_gap_win_5|safe }}
    {{ fig_gap_win_10|safe }}
  </div>

  <div class="section">
    <h2>Win rate vs distance to AVWAP</h2>
    {{ fig_avwap_win_5|safe }}
    {{ fig_avwap_win_10|safe }}
  </div>

  <div class="section">
    <h2>Gap size vs 10d retrace %</h2>
    {{ fig_scatter|safe }}
  </div>

  <div class="section">
    <h2>SPY vs 10d retrace rate (by day)</h2>
    {{ fig_spy|safe }}
  </div>

  <div class="section">
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

    # Daily summary
    grp = df.groupby('date') if 'date' in df.columns else None
    if grp is not None:
        # Use named aggregation with output column names
        agg_kwargs = {'total': ('ticker', 'count')}
        if retraced5_col:
            agg_kwargs['retraced5'] = (retraced5_col, 'sum')
        if retraced10_col:
            agg_kwargs['retraced10'] = (retraced10_col, 'sum')
        daily = grp.agg(**agg_kwargs).reset_index()
        if 'retraced5' in daily.columns:
            daily['rate5'] = 100 * daily['retraced5'] / daily['total']
        else:
            daily['rate5'] = np.nan
        if 'retraced10' in daily.columns:
            daily['rate10'] = 100 * daily['retraced10'] / daily['total']
        else:
            daily['rate10'] = np.nan
    else:
        daily = pd.DataFrame(columns=['date', 'total', 'retraced5', 'retraced10', 'rate5', 'rate10'])

    # Performance summary (overall and by year)
    # Define columns to use
    win_col_10 = retraced10_col
    win_col_5 = retraced5_col
    retrace_pct_10_col = 'retrace_percentage_10d' if 'retrace_percentage_10d' in df.columns else ('retrace_percentage' if 'retrace_percentage' in df.columns else None)
    retrace_pct_5_col = 'retrace_percentage_5d' if 'retrace_percentage_5d' in df.columns else None

    # Build fractional retrace columns for PF computation
    if retrace_pct_10_col:
        df['retrace_frac_10d'] = pd.to_numeric(df[retrace_pct_10_col], errors='coerce') / 100.0
        df['retrace_frac_10d'] = df['retrace_frac_10d'].clip(lower=0.0, upper=1.0).fillna(0.0)
        profit_units_total = float(df['retrace_frac_10d'].sum())
        loss_units_total = float((1.0 - df['retrace_frac_10d']).sum())
        overall_profit_factor = ("∞" if profit_units_total > 0 and loss_units_total == 0
                                 else (f"{(profit_units_total / loss_units_total):.2f}" if loss_units_total > 0 else "0.00"))
    else:
        df['retrace_frac_10d'] = np.nan
        overall_profit_factor = "n/a"

    if retrace_pct_5_col:
        df['retrace_frac_5d'] = pd.to_numeric(df[retrace_pct_5_col], errors='coerce') / 100.0
        df['retrace_frac_5d'] = df['retrace_frac_5d'].clip(lower=0.0, upper=1.0).fillna(0.0)
    else:
        df['retrace_frac_5d'] = np.nan

    # Overall win rate (10d-based to match prior behavior)
    if win_col_10 and win_col_10 in df.columns and len(df) > 0:
        overall_win_rate = f"{(100.0 * pd.to_numeric(df[win_col_10], errors='coerce').fillna(0).astype(float).mean()):.2f}"
    else:
        overall_win_rate = "n/a"

    # Yearly segmentation tables for 5d and 10d
    if 'date' in df.columns and not df['date'].isna().all():
        df_year = df.copy()
        df_year['year'] = df_year['date'].dt.year

        # 5d table
        rows5 = []
        for y, g in df_year.groupby('year', dropna=True):
            trades = int(len(g))
            if win_col_5 and win_col_5 in g.columns and trades > 0:
                wr5 = 100.0 * pd.to_numeric(g[win_col_5], errors='coerce').fillna(0).astype(float).mean()
            else:
                wr5 = np.nan
            if 'retrace_frac_5d' in g.columns and not g['retrace_frac_5d'].isna().all():
                p5 = float(g['retrace_frac_5d'].sum())
                l5 = float((1.0 - g['retrace_frac_5d']).sum())
                pf5 = np.inf if p5 > 0 and l5 == 0 else (p5 / l5 if l5 > 0 else np.nan)
            else:
                pf5 = np.nan

            # Averages for 5d MFE/MAE percent
            if 'mfe_percent_5d' in g.columns:
                avg_mfe5p = pd.to_numeric(g['mfe_percent_5d'], errors='coerce').mean()
            else:
                avg_mfe5p = np.nan
            if 'mae_percent_5d' in g.columns:
                avg_mae5p = pd.to_numeric(g['mae_percent_5d'], errors='coerce').mean()
            else:
                avg_mae5p = np.nan

            rows5.append({
                'Year': int(y),
                'Trades': trades,
                'Win Rate %': round(wr5, 2) if pd.notna(wr5) else np.nan,
                'Profit Factor': (round(pf5, 2) if np.isfinite(pf5) else (np.nan if np.isnan(pf5) else np.inf)),
                'Avg MFE %': round(avg_mfe5p, 2) if pd.notna(avg_mfe5p) else np.nan,
                'Avg MAE %': round(avg_mae5p, 2) if pd.notna(avg_mae5p) else np.nan
            })
        perf_df_5 = pd.DataFrame(rows5).sort_values('Year') if rows5 else pd.DataFrame(columns=['Year', 'Trades', 'Win Rate %', 'Profit Factor'])
        perf_table_5_html = perf_df_5.to_html(index=False)

        # 10d table
        rows10 = []
        for y, g in df_year.groupby('year', dropna=True):
            trades = int(len(g))
            if win_col_10 and win_col_10 in g.columns and trades > 0:
                wr10 = 100.0 * pd.to_numeric(g[win_col_10], errors='coerce').fillna(0).astype(float).mean()
            else:
                wr10 = np.nan
            if 'retrace_frac_10d' in g.columns and not g['retrace_frac_10d'].isna().all():
                p10 = float(g['retrace_frac_10d'].sum())
                l10 = float((1.0 - g['retrace_frac_10d']).sum())
                pf10 = np.inf if p10 > 0 and l10 == 0 else (p10 / l10 if l10 > 0 else np.nan)
            else:
                pf10 = np.nan

            # Averages for 10d MFE/MAE percent
            if 'mfe_percent_10d' in g.columns:
                avg_mfe10p = pd.to_numeric(g['mfe_percent_10d'], errors='coerce').mean()
            else:
                avg_mfe10p = np.nan
            if 'mae_percent_10d' in g.columns:
                avg_mae10p = pd.to_numeric(g['mae_percent_10d'], errors='coerce').mean()
            else:
                avg_mae10p = np.nan

            rows10.append({
                'Year': int(y),
                'Trades': trades,
                'Win Rate %': round(wr10, 2) if pd.notna(wr10) else np.nan,
                'Profit Factor': (round(pf10, 2) if np.isfinite(pf10) else (np.nan if np.isnan(pf10) else np.inf)),
                'Avg MFE %': round(avg_mfe10p, 2) if pd.notna(avg_mfe10p) else np.nan,
                'Avg MAE %': round(avg_mae10p, 2) if pd.notna(avg_mae10p) else np.nan
            })
        perf_df_10 = pd.DataFrame(rows10).sort_values('Year') if rows10 else pd.DataFrame(columns=['Year', 'Trades', 'Win Rate %', 'Profit Factor'])
        perf_table_10_html = perf_df_10.to_html(index=False)
    else:
        perf_table_5_html = pd.DataFrame(columns=['Year', 'Trades', 'Win Rate %', 'Profit Factor']).to_html(index=False)
        perf_table_10_html = pd.DataFrame(columns=['Year', 'Trades', 'Win Rate %', 'Profit Factor']).to_html(index=False)

    # Charts
    fig_daily_5 = px.line(daily, x='date', y='rate5', title='Daily 5d retracement rate (%)') if not daily.empty else px.scatter(title='No daily data')
    fig_daily_10 = px.line(daily, x='date', y='rate10', title='Daily 10d retracement rate (%)') if not daily.empty else px.scatter(title='No daily data')

    # Monthly win rate charts (5d and 10d)
    if 'date' in df.columns and not df['date'].isna().all():
        df_m = df.copy()
        df_m['month'] = df_m['date'].dt.to_period('M').dt.to_timestamp()
        monthly_5 = None
        monthly_10 = None
        if retraced5_col:
            df_m['retr5_num'] = pd.to_numeric(df_m[retraced5_col], errors='coerce').fillna(0).astype(float)
            monthly_5 = (df_m.groupby('month', dropna=True)['retr5_num'].mean().reset_index(name='rate'))
            monthly_5['rate'] = 100 * monthly_5['rate']
            fig_monthly_5 = px.bar(monthly_5, x='month', y='rate', title='Monthly 5d win rate (%)')
        else:
            fig_monthly_5 = px.scatter(title='Monthly 5d win rate (n/a)')
        if retraced10_col:
            df_m['retr10_num'] = pd.to_numeric(df_m[retraced10_col], errors='coerce').fillna(0).astype(float)
            monthly_10 = (df_m.groupby('month', dropna=True)['retr10_num'].mean().reset_index(name='rate'))
            monthly_10['rate'] = 100 * monthly_10['rate']
            fig_monthly_10 = px.bar(monthly_10, x='month', y='rate', title='Monthly 10d win rate (%)')
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
        fig_bucket = px.bar(bucket, x='gap_bucket', y='rate10', title='10d retracement rate by gap bucket (%)')
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
            fig_gap_win_5 = px.bar(gap_win_5, x='gap_bucket', y='rate', title='Win rate vs gap size (5d, %)')
        else:
            fig_gap_win_5 = px.scatter(title='Win rate vs gap size (5d unavailable)')
        if retraced10_col:
            gap_win_10 = (gb['win10'].mean().reset_index(name='rate'))
            gap_win_10['rate'] = 100 * gap_win_10['rate']
            fig_gap_win_10 = px.bar(gap_win_10, x='gap_bucket', y='rate', title='Win rate vs gap size (10d, %)')
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
            fig_avwap_win_5 = px.bar(avwap_win_5, x='dist_bucket', y='rate', title='Win rate vs distance to AVWAP (5d, %)')
        else:
            fig_avwap_win_5 = px.scatter(title='Win rate vs distance to AVWAP (5d unavailable)')
        if retraced10_col:
            avwap_win_10 = (gv['win10'].mean().reset_index(name='rate'))
            avwap_win_10['rate'] = 100 * avwap_win_10['rate']
            fig_avwap_win_10 = px.bar(avwap_win_10, x='dist_bucket', y='rate', title='Win rate vs distance to AVWAP (10d, %)')
        else:
            fig_avwap_win_10 = px.scatter(title='Win rate vs distance to AVWAP (10d unavailable)')
    else:
        import plotly.graph_objects as go
        fig_avwap_win_5 = go.Figure().update_layout(title='Win rate vs distance to AVWAP (5d) — no data')
        fig_avwap_win_10 = go.Figure().update_layout(title='Win rate vs distance to AVWAP (10d) — no data')

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
        overall_win_rate=overall_win_rate,
        overall_profit_factor=overall_profit_factor,
        perf_table_5=perf_table_5_html,
        perf_table_10=perf_table_10_html,
        fig_daily_5=fig_daily_5.to_html(include_plotlyjs='cdn', full_html=False),
        fig_daily_10=fig_daily_10.to_html(include_plotlyjs=False, full_html=False),
        fig_monthly_5=fig_monthly_5.to_html(include_plotlyjs=False, full_html=False),
        fig_monthly_10=fig_monthly_10.to_html(include_plotlyjs=False, full_html=False),
        fig_monthly_both=fig_monthly_both.to_html(include_plotlyjs=False, full_html=False),
        fig_gap_hist=fig_gap_hist.to_html(include_plotlyjs=False, full_html=False),
        fig_bucket=fig_bucket.to_html(include_plotlyjs=False, full_html=False),
        fig_gap_win_5=fig_gap_win_5.to_html(include_plotlyjs=False, full_html=False),
        fig_gap_win_10=fig_gap_win_10.to_html(include_plotlyjs=False, full_html=False),
        fig_avwap_win_5=fig_avwap_win_5.to_html(include_plotlyjs=False, full_html=False),
        fig_avwap_win_10=fig_avwap_win_10.to_html(include_plotlyjs=False, full_html=False),
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
