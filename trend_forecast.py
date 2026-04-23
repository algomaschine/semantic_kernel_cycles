#!/usr/bin/env python3
"""
STRATEGIC TREND ANALYSIS – YEARLY SEASONALITY ONLY
Based on the working PDF script, converted to interactive HTML.
- Same forecasting logic (additive seasonality, yearly component)
- Prints curve values to console for verification
- Generates HTML with zoomable cluster plots and weekly heatmap
"""

import time
import random
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
from prophet import Prophet
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import argparse

# ========================= CONFIGURATION =========================
DEFAULT_BASE_KEYWORD = 'love'
GEO = 'EN'                     # worldwide English (use '' for any language)
TIMEFRAME = 'today 5-y'
HL = 'en-US'
CURRENT_YEAR = datetime.now().year
YEAR_TO_FORECAST = CURRENT_YEAR
AUTHOR_NAME = "Eduard Samokhvalov"
AUTHOR_TITLE = "Quantitative Developer"
CONTACT_INFO = "edward.samokhvalov@gmail.com | Telegram: @EduardSam"

DATA_DIR = "data"

# ==================== HELPER FUNCTIONS (IDENTICAL TO WORKING SCRIPT) ====================
def create_retry_session(retries=5, backoff_factor=1):
    session = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries,
                  backoff_factor=backoff_factor,
                  allowed_methods=["GET", "POST"],
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def expand_keywords(base_list, max_suggestions=3):
    expanded = set(base_list)
    session = create_retry_session()
    for word in base_list:
        print(f"Expanding: {word}...")
        try:
            resp = session.get(f'https://api.datamuse.com/words?rel_trg={word}&max=5', timeout=15)
            if resp.status_code == 200:
                for res in resp.json():
                    expanded.add(res['word'])
        except Exception as e:
            print(f"  DataMuse error: {e}")
        time.sleep(random.uniform(2, 4))
        try:
            pytrends = TrendReq(hl=HL, tz=360, timeout=30, requests_args={'verify': True})
            suggestions = pytrends.suggestions(word)
            for s in suggestions[:max_suggestions]:
                if 'title' in s:
                    expanded.add(s['title'])
        except Exception as e:
            print(f"  Google Suggestions error: {e}")
        time.sleep(random.uniform(3, 5))
    return list(expanded)

def fetch_trend_data(keyword, base_keyword, timeframe, geo, retries=3):
    keyword_dir = os.path.join(DATA_DIR, base_keyword)
    os.makedirs(keyword_dir, exist_ok=True)
    filename = os.path.join(keyword_dir, f"{keyword.replace(' ', '_')}.csv")
    if os.path.exists(filename):
        try:
            df_cache = pd.read_csv(filename, parse_dates=['ds'])
            if not df_cache.empty:
                print(f"    Using cached data from {filename}")
                return df_cache[['ds', 'y']].copy()
        except Exception as e:
            print(f"    Error reading cache: {e}. Re-downloading.")
    for attempt in range(retries):
        try:
            pytrends = TrendReq(hl=HL, tz=360, timeout=45, requests_args={'verify': True})
            pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if df.empty or keyword not in df.columns:
                print(f"  No data for '{keyword}' (attempt {attempt+1})")
                return None
            df_out = df.reset_index()[['date', keyword]].rename(columns={'date': 'ds', keyword: 'y'})
            df_out['ds'] = pd.to_datetime(df_out['ds'])
            df_out = df_out.sort_values('ds')
            df_out.to_csv(filename, index=False)
            print(f"    Saved to {filename}")
            return df_out
        except Exception as e:
            print(f"  Error fetching '{keyword}': {e} (attempt {attempt+1}/{retries})")
            time.sleep(10 * (attempt + 1))
    return None

# ==================== FORECASTING (SAME AS WORKING PDF SCRIPT) ====================
def forecast_yearly_seasonality(df, keyword, forecast_year):
    """
    Exactly as in the working PDF script:
    - Fit Prophet on all data (no filtering)
    - Forecast 450 days
    - Extract 'yearly' component for forecast_year
    - Normalise by absolute max
    - Detect peaks
    """
    # Use additive seasonality (as you requested)
    model = Prophet(seasonality_mode='additive', yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=450, freq='D')
    forecast = model.predict(future)
    f_year = forecast[forecast['ds'].dt.year == forecast_year]
    if f_year.empty:
        return None, None, []
    dates = f_year['ds'].values
    yearly = f_year['yearly'].values
    # Normalise
    if np.max(np.abs(yearly)) > 0:
        norm_yearly = yearly / np.max(np.abs(yearly))
    else:
        norm_yearly = yearly
    # Peaks
    peaks, _ = find_peaks(norm_yearly, prominence=0.1)
    peak_dates = [pd.to_datetime(dates[i]) for i in peaks]
    return dates, norm_yearly, peak_dates

# ==================== WEEKLY HEATMAP (AVERAGE INFLUENCE) ====================
def create_weekly_heatmap(all_curves, all_dates, keywords):
    df_all = pd.DataFrame(index=all_dates)
    for i, kw in enumerate(keywords):
        df_all[kw] = all_curves[i]
    df_all['avg'] = df_all.mean(axis=1)
    df_all['week_start'] = df_all.index - pd.to_timedelta(df_all.index.dayofweek, unit='d')
    df_all['day_name'] = df_all.index.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_all['week_label'] = df_all['week_start'].dt.strftime('%b %d') + ' – ' + (df_all['week_start'] + pd.Timedelta(days=6)).dt.strftime('%b %d')
    pivot = df_all.pivot_table(index='week_start', columns='day_name', values='avg', aggfunc='mean')
    pivot = pivot[day_order]
    weeks = sorted(pivot.index)
    pivot = pivot.reindex(weeks).fillna(0)
    fig = px.imshow(pivot,
                    labels=dict(x="Day of week", y="Week (Monday – Sunday)", color="Avg influence"),
                    title="Weekly Heatmap – Average Yearly Influence",
                    color_continuous_scale=['white', 'red'],
                    aspect='auto')
    fig.update_layout(height=600, width=None, autosize=True, margin=dict(l=140))
    fig.update_yaxes(tickvals=list(range(len(weeks))), ticktext=[w.strftime('%b %d') + ' – ' + (w+timedelta(days=6)).strftime('%b %d') for w in weeks])
    return fig

# ==================== MAIN PIPELINE ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-ready', action='store_true')
    parser.add_argument('--base-keyword', type=str, default=DEFAULT_BASE_KEYWORD)
    args = parser.parse_args()
    base_keyword = args.base_keyword
    data_ready = args.data_ready

    print("=== STRATEGIC TREND ANALYSIS (YEARLY SEASONALITY) ===")
    print(f"Base keyword: {base_keyword}")
    print(f"Forecast year: {YEAR_TO_FORECAST}\n")

    # --- Get keywords and data (identical to working script) ---
    if not data_ready:
        full_keywords = expand_keywords([base_keyword])
        print(f"\nSemantic core: {full_keywords}\n")
        all_data = {}
        for kw in full_keywords:
            print(f"Processing: {kw}...")
            time.sleep(random.uniform(20, 35))
            df = fetch_trend_data(kw, base_keyword, TIMEFRAME, GEO)
            if df is not None:
                all_data[kw] = df
                print(f"  Data ready.\n")
            else:
                print(f"  Skipping.\n")
    else:
        keyword_dir = os.path.join(DATA_DIR, base_keyword)
        if not os.path.isdir(keyword_dir):
            print(f"ERROR: {keyword_dir} not found. Run without --data-ready first.")
            sys.exit(1)
        all_data = {}
        for f in os.listdir(keyword_dir):
            if f.endswith('.csv'):
                kw = os.path.splitext(f)[0].replace('_', ' ')
                df = pd.read_csv(os.path.join(keyword_dir, f), parse_dates=['ds'])
                if 'ds' in df and 'y' in df:
                    all_data[kw] = df
                    print(f"  Loaded {kw}")
        if not all_data:
            print("No data loaded.")
            sys.exit(1)

    # --- Forecast yearly seasonality (SAME AS WORKING SCRIPT) ---
    all_yearly = {}
    all_peaks = []
    all_curves = []

    for kw, df in all_data.items():
        print(f"  Forecasting {kw}...")
        dates, norm_yearly, peaks = forecast_yearly_seasonality(df, kw, YEAR_TO_FORECAST)
        if dates is None:
            print(f"    Failed for {kw}")
            continue
        all_yearly[kw] = (dates, norm_yearly)
        all_curves.append(norm_yearly)
        for p in peaks:
            all_peaks.append({'date': p, 'keyword': kw})
        # DEBUG: print first few values of the curve to show it's not a straight line
        print(f"    First 5 yearly values for {kw}: {norm_yearly[:5]}")
        print(f"    Peaks found: {len(peaks)}\n")

    if not all_yearly:
        print("No yearly seasonality extracted. Exiting.")
        return

    # --- Clustering (same as working script) ---
    matrix = np.array([v[1] for v in all_yearly.values()])
    n_clusters = min(3, len(all_yearly))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)
    keyword_list = list(all_yearly.keys())
    first_dates = next(iter(all_yearly.values()))[0]

    # --- Generate HTML report ---
    print("\nGenerating interactive HTML report with zoomable plots...")
    html_parts = []

    html_parts.append(f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<title>Yearly Seasonality – {base_keyword.upper()} {YEAR_TO_FORECAST}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
    body {{ font-family: Arial; margin:20px; background:#f5f5f5; }}
    .container {{ max-width:1400px; margin:auto; background:white; padding:20px; border-radius:8px; }}
    .annotation {{ background:#eef; padding:15px; border-left:5px solid #3498db; margin:20px 0; }}
    .plot {{ margin:40px 0; width:100%; overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; margin:20px 0; }}
    th,td {{ border:1px solid #ddd; padding:8px; text-align:center; }}
    th {{ background:#3498db; color:white; }}
</style>
</head>
<body><div class="container">
<h1>Yearly Seasonal Patterns – {base_keyword.upper()} ({YEAR_TO_FORECAST})</h1>
<p>Additive seasonality, normalised to [-1,1]. Triangles = peaks. Hover for exact values.</p>
<div class="annotation"><strong>Interactive report:</strong> Zoom with mouse wheel, pan, hover for details.</div>
""")

    # Cluster plots with Plotly
    for c in range(n_clusters):
        cluster_kws = [keyword_list[i] for i, lbl in enumerate(labels) if lbl == c]
        fig = go.Figure()
        for kw in cluster_kws:
            dates, values = all_yearly[kw]
            fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name=kw,
                                     hovertemplate='%{{x|%Y-%m-%d}}<br>%{{y:.3f}}<extra></extra>'))
            # Peaks for this keyword
            kw_peaks = [p['date'] for p in all_peaks if p['keyword'] == kw]
            if kw_peaks:
                peak_vals = [values[list(dates).index(d)] for d in kw_peaks]
                fig.add_trace(go.Scatter(x=kw_peaks, y=peak_vals, mode='markers',
                                         marker=dict(symbol='triangle-up', size=10, color='red'),
                                         name=f'{kw} peaks', showlegend=False,
                                         hovertemplate='Peak: %{{x|%Y-%m-%d}}<br>%{{y:.3f}}<extra></extra>'))
        fig.update_layout(title=f'Cluster {c+1} – Yearly Seasonality', height=500,
                          xaxis_title='Date', yaxis_title='Normalised influence')
        html_parts.append(f'<div class="plot"><h3>Cluster {c+1}</h3><div id="cluster{c}"></div></div>')
        html_parts.append(f'<script>Plotly.newPlot("cluster{c}", {fig.to_json()});</script>')

    # Weekly heatmap (average influence)
    ordered_curves = [all_yearly[kw][1] for kw in keyword_list]
    hm_fig = create_weekly_heatmap(ordered_curves, first_dates, keyword_list)
    html_parts.append(f'<div class="plot"><h3>Weekly Heatmap (Average Influence)</h3><div id="heatmap"></div></div>')
    html_parts.append(f'<script>Plotly.newPlot("heatmap", {hm_fig.to_json()});</script>')

    # Keyword vs Month table
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    peak_mat = pd.DataFrame('', index=keyword_list, columns=month_names)
    for kw in keyword_list:
        kw_peaks = [p['date'] for p in all_peaks if p['keyword'] == kw]
        for p in kw_peaks:
            m = month_names[p.month-1]
            d = p.strftime('%d/%m')
            cur = peak_mat.loc[kw, m]
            peak_mat.loc[kw, m] = f"{cur}, {d}" if cur else d
    html_parts.append(f'<div class="plot">{peak_mat.to_html(escape=False)}</div>')

    # All peaks table
    if all_peaks:
        pdf = pd.DataFrame(all_peaks).sort_values('date')
        pdf['date'] = pdf['date'].dt.strftime('%Y-%m-%d')
        html_parts.append(f'<div class="plot">{pdf.to_html(index=False)}</div>')

    html_parts.append(f'<div class="footer">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>')
    html_parts.append('</div></body></html>')

    out_file = f"./Strategic_Trend_Report_{base_keyword}_{YEAR_TO_FORECAST}.html"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_parts))

    print(f"\n✅ HTML report saved: {out_file}")
    print(f"   Keywords: {len(all_yearly)} | Clusters: {n_clusters} | Total peaks: {len(all_peaks)}")
    print("=== DONE ===")

if __name__ == "__main__":
    main()
