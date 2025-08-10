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
    <h2>Daily retracement rates</h2>
    {{ fig_daily_5|safe }}
    {{ fig_daily_10|safe }}
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
    df = pd.read_csv(csv_path)

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

    # Charts
    fig_daily_5 = px.line(daily, x='date', y='rate5', title='Daily 5d retracement rate (%)') if not daily.empty else px.scatter(title='No daily data')
    fig_daily_10 = px.line(daily, x='date', y='rate10', title='Daily 10d retracement rate (%)') if not daily.empty else px.scatter(title='No daily data')

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
        fig_daily_5=fig_daily_5.to_html(include_plotlyjs='cdn', full_html=False),
        fig_daily_10=fig_daily_10.to_html(include_plotlyjs=False, full_html=False),
        fig_gap_hist=fig_gap_hist.to_html(include_plotlyjs=False, full_html=False),
        fig_bucket=fig_bucket.to_html(include_plotlyjs=False, full_html=False),
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
