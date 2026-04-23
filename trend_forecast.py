#!/usr/bin/env python3
"""
Full Strategic Analysis: Clustering + Heatmap + Tables (Daily, Lightweight, Full-width)
Uses .to_html() for plots – same reliable method as the working overlay.
"""

import time
import random
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytrends.request import TrendReq
from prophet import Prophet
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go
import plotly.express as px
import os
import argparse

# ========================= CONFIGURATION =========================
DEFAULT_BASE_KEYWORD = 'love'
GEO = 'EN'
TIMEFRAME = 'today 7-y'
HL = 'en-US'
CURRENT_YEAR = datetime.now().year
FORECAST_YEAR = CURRENT_YEAR
DATA_DIR = "data"
AUTHOR_NAME = "Eduard Samokhvalov"
AUTHOR_TITLE = "Quantitative Developer"
CONTACT_INFO = "edward.samokhvalov@gmail.com | Telegram: @EduardSam"

# ==================== RETRY SESSION ====================
session = requests.Session()
retry = Retry(connect=5, read=5, status=5, backoff_factor=2, allowed_methods=None)
adapter = HTTPAdapter(max_retries=retry)
session.mount('https://', adapter)
pytrends = TrendReq(hl=HL, tz=360, timeout=30, requests_args={'verify': True})

# ==================== EXPANSION ====================
def expand_keywords(base_list, max_suggestions=3):
    expanded = set(base_list)
    for word in base_list:
        try:
            print(f"Expanding: {word}...")
            response = requests.get(f'https://api.datamuse.com/words?ml={word}&max=10', timeout=15)
            if response.status_code == 200:
                for res in response.json():
                    expanded.add(res['word'])
            time.sleep(random.uniform(5, 10))
            try:
                suggestions = pytrends.suggestions(word)
                for s in suggestions[:max_suggestions]:
                    if 'title' in s:
                        expanded.add(s['title'])
            except:
                print(f"Google suggestions failed for {word}")
            time.sleep(random.uniform(5, 10))
        except Exception as e:
            print(f"Expansion error: {e}")
    return list(expanded)

# ==================== DATA FETCHING (with caching) ====================
def fetch_trend_data(keyword, base_keyword, timeframe, geo):
    keyword_dir = os.path.join(DATA_DIR, base_keyword)
    os.makedirs(keyword_dir, exist_ok=True)
    filename = os.path.join(keyword_dir, f"{keyword.replace(' ', '_')}.csv")
    if os.path.exists(filename):
        try:
            df_cache = pd.read_csv(filename, parse_dates=['ds'])
            if not df_cache.empty:
                print(f"    Using cached data from {filename}")
                return df_cache[['ds', 'y']].copy()
        except:
            pass
    print(f"    Downloading fresh data for {keyword}...")
    pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df.empty or keyword not in df.columns:
        return None
    df_out = df.reset_index()[['date', keyword]].rename(columns={'date': 'ds', keyword: 'y'})
    df_out['ds'] = pd.to_datetime(df_out['ds'])
    df_out = df_out.sort_values('ds')
    df_out.to_csv(filename, index=False)
    return df_out

# ==================== FORECAST (daily, rounded, identical to working overlay) ====================
def forecast_yearly_seasonality(df, keyword, forecast_year):
    m = Prophet(seasonality_mode='additive', yearly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=365, freq='D')
    forecast = m.predict(future)
    f_year = forecast[forecast['ds'].dt.year == forecast_year]
    if f_year.empty:
        return None, None, []
    dates = f_year['ds'].values
    yearly = f_year['yearly'].values
    if np.max(np.abs(yearly)) > 0:
        norm_yearly = yearly / np.max(np.abs(yearly))
    else:
        norm_yearly = yearly
    norm_yearly = np.round(norm_yearly, 4)
    peaks, _ = find_peaks(norm_yearly, prominence=0.1)
    peak_dates = [pd.to_datetime(dates[i]) for i in peaks]
    return dates, norm_yearly, peak_dates

# ==================== WEEKLY HEATMAP ====================
def create_weekly_heatmap(all_curves, all_dates, keywords):
    import plotly.graph_objects as go
    from datetime import timedelta

    df_all = pd.DataFrame(index=all_dates)
    for i, kw in enumerate(keywords):
        df_all[kw] = all_curves[i]
    df_all['avg'] = df_all.mean(axis=1)
    df_all['week_start'] = df_all.index - pd.to_timedelta(df_all.index.dayofweek, unit='d')
    df_all['day_name'] = df_all.index.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = df_all.pivot_table(index='week_start', columns='day_name', values='avg', aggfunc='mean')
    pivot = pivot[day_order]
    weeks = sorted(pivot.index)
    pivot = pivot.reindex(weeks).fillna(0)

    # Data matrix
    z = pivot.values
    # Annotations for Monday cells (first column)
    annotations = []
    for i, week in enumerate(weeks):
        week_label = week.strftime('%b %d') + ' – ' + (week + timedelta(days=6)).strftime('%b %d')
        annotations.append(dict(
            x='Monday', y=weeks[i], text=week_label,
            xref='x', yref='y', showarrow=False,
            font=dict(size=9), xanchor='center', yanchor='middle'
        ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=day_order,
        y=weeks,
        colorscale=['white', 'red'],
        hoverongaps=False,
        hovertemplate='Week: %{y}<br>Day: %{x}<br>Influence: %{z:.3f}<extra></extra>'
    ))

    for ann in annotations:
        fig.add_annotation(ann)

    # Dynamic height: at least 25px per week row
    row_height = 28
    fig_height = max(500, len(weeks) * row_height + 80)

    fig.update_layout(
        title="Weekly Activity Heatmap (average normalized influence across all keywords)",
        height=fig_height,
        width=None,
        autosize=True,
        margin=dict(l=20, r=20, t=80, b=40),
        yaxis=dict(showticklabels=False, autorange='reversed'),
        xaxis=dict(tickangle=0, side='top')
    )
    return fig

# ==================== SILHOUETTE CLUSTERING ====================
def find_optimal_clusters(matrix, max_clusters=6):
    n_samples = matrix.shape[0]
    if n_samples < 2:
        return 1
    best_n = 2
    best_score = -1
    max_possible = min(max_clusters, n_samples - 1)
    for n in range(2, max_possible + 1):
        if n >= n_samples:
            break
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_score = score
            best_n = n
    return best_n if best_score > 0 else 2

# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-keyword', type=str, default=DEFAULT_BASE_KEYWORD)
    parser.add_argument('--data-ready', action='store_true')
    args = parser.parse_args()
    base_keyword = args.base_keyword
    data_ready = args.data_ready

    print(f"=== Full Analysis: Clustering + Heatmap + Tables (Daily) ===")
    print(f"Base: {base_keyword}, Forecast year: {FORECAST_YEAR}\n")

    # --- Get data ---
    if not data_ready:
        full_keywords = expand_keywords([base_keyword])
        print(f"\nSemantic core: {full_keywords}\n")
        all_data = {}
        for kw in full_keywords:
            print(f"Processing: {kw}...")
            time.sleep(random.uniform(40, 60))
            df = fetch_trend_data(kw, base_keyword, TIMEFRAME, GEO)
            if df is not None:
                all_data[kw] = df
                print(f"  Ready.\n")
            else:
                print(f"  Skipping.\n")
    else:
        keyword_dir = os.path.join(DATA_DIR, base_keyword)
        if not os.path.isdir(keyword_dir):
            print(f"ERROR: {keyword_dir} not found. Run without --data-ready first.")
            return
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
            return

    # --- Forecast (store original curves) ---
    yearly_data = {}
    all_peaks = []
    all_curves = []
    for kw, df in all_data.items():
        print(f"  Forecasting {kw}...")
        dates, norm_yearly, peaks = forecast_yearly_seasonality(df, kw, FORECAST_YEAR)
        if dates is not None:
            yearly_data[kw] = (dates, norm_yearly, peaks)
            all_curves.append(norm_yearly)
            for p in peaks:
                all_peaks.append({'date': p, 'keyword': kw})
            print(f"    First 5 values: {norm_yearly[:5]}")
            print(f"    Peaks: {len(peaks)}\n")
        else:
            print(f"    Failed.\n")

    if not yearly_data:
        print("No seasonal curves extracted.")
        return

    # --- Prepare matrix for clustering (use copies) ---
    keyword_list = list(yearly_data.keys())
    first_dates = yearly_data[keyword_list[0]][0]
    matrix = []
    for kw in keyword_list:
        _, curve, _ = yearly_data[kw]
        matrix.append(curve.copy())
    matrix = np.array(matrix)

    # --- Clustering ---
    n_clusters = find_optimal_clusters(matrix)
    print(f"Optimal clusters: {n_clusters}")
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)
    else:
        labels = np.zeros(len(keyword_list), dtype=int)

    # --- Generate HTML report using .to_html() for plots ---
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Strategic Trend Forecast {FORECAST_YEAR} – {base_keyword.upper()}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #1a237e; }}
        .annotation {{ background: #e8eaf6; padding: 15px; border-left: 5px solid #3f51b5; margin: 20px 0; border-radius: 4px; }}
        .plot {{ margin: 40px 0; width: 100%; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
        th {{ background: #3f51b5; color: white; }}
        tr:nth-child(even) {{ background-color: #f5f5f5; }}
        .footer {{ text-align: center; margin-top: 40px; font-size: 0.8em; color: #666; border-top: 1px solid #ddd; padding-top: 20px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Strategic Trend Forecast {FORECAST_YEAR}</h1>
    <p><strong>Base keyword:</strong> {base_keyword.upper()} | <strong>Data:</strong> Google Trends (worldwide English) | <strong>Forecast year:</strong> {FORECAST_YEAR}</p>
    <p><strong>Prepared by:</strong> {AUTHOR_NAME} ({AUTHOR_TITLE})<br>{CONTACT_INFO}</p>
    <div class="annotation">
        <strong>📘 How to read this report:</strong><br>
        • <strong>Cluster plots</strong> show daily normalised yearly influence (additive seasonality). Triangles = predicted peaks.<br>
        • <strong>Hover</strong> over any line to see exact date and influence. Zoom & pan with mouse.<br>
        • <strong>Weekly heatmap</strong> – average influence per weekday per week (white→red).<br>
        • <strong>Keyword vs Month table</strong> – exact peak dates (DD/MM).<br>
        • <strong>Peak dates table</strong> – all peaks sorted chronologically.<br>
        • Number of clusters determined automatically by silhouette score.
    </div>
""")

    # --- Cluster plots: generate HTML using .to_html() (reliable) ---
    cluster_htmls = []
    for c in range(n_clusters):
        cluster_kws = [keyword_list[i] for i, lbl in enumerate(labels) if lbl == c]
        if not cluster_kws:
            continue
        fig = go.Figure()
        for kw in cluster_kws:
            dates, curve, peaks = yearly_data[kw]
            fig.add_trace(go.Scattergl(
                x=dates, y=curve, mode='lines', name=kw,
                line=dict(width=1.2),
                hovertemplate='%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>'
            ))
            if peaks:
                peak_vals = [curve[list(dates).index(p)] for p in peaks]
                fig.add_trace(go.Scatter(
                    x=peaks, y=peak_vals, mode='markers',
                    marker=dict(symbol='triangle-up', size=8, color='red'),
                    name=f'{kw} peaks', showlegend=False,
                    hovertemplate='Peak: %{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>'
                ))
        fig.update_layout(
            title=f'Cluster {c+1} (size: {len(cluster_kws)})',
            xaxis_title='Date',
            yaxis_title='Normalised influence',
            hovermode='closest',
            height=500,
            width=None,
            autosize=True
        )
        # Generate HTML fragment for this plot
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
        cluster_htmls.append((c+1, plot_html))

    # Append cluster plots in order
    for c_num, plot_html in cluster_htmls:
        html_parts.append(f'<div class="plot"><h3>Cluster {c_num}</h3>{plot_html}</div>')

    # --- Weekly heatmap HTML ---
    heatmap_fig = create_weekly_heatmap(all_curves, first_dates, keyword_list)
    heatmap_fig.update_layout(width=None, autosize=True, title=None)  # title already in the figure
    heatmap_html = heatmap_fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
    html_parts.append(f'<div class="plot"><h3>Weekly Activity Heatmap (Average Influence)</h3>{heatmap_html}</div>')

    # --- Keyword vs Month peak table ---
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

    # --- All peaks table ---
    if all_peaks:
        peak_df = pd.DataFrame(all_peaks).sort_values('date')
        peak_df['date'] = peak_df['date'].dt.strftime('%Y-%m-%d')
        html_parts.append(f'<div class="plot">{peak_df.to_html(index=False)}</div>')

    html_parts.append(f'<div class="footer">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>Prophet additive yearly component | Optimal clusters: {n_clusters} (silhouette)</div>')
    html_parts.append('</div></body></html>')

    safe_keyword = base_keyword.replace(' ', '_')
    out_file = f"./Full_Analysis_{safe_keyword}_{FORECAST_YEAR}.html"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_parts))

    print(f"\n✅ Full analysis report saved: {out_file}")
    print(f"   Keywords: {len(yearly_data)} | Clusters: {n_clusters} | Total peaks: {len(all_peaks)}")
    print("=== Done ===")

if __name__ == "__main__":
    main()
