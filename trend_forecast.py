#!/usr/bin/env python3
"""
Strategic Trend Analysis – Yearly Seasonality Only (Additive)
- Uses only Prophet's 'yearly' component
- Normalized to [-1,1]
- Interactive cluster plots (hover, zoom)
- Weekly heatmap (average across keywords)
"""

import os, sys, time, random, requests, argparse
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

# ========================= CONFIG =========================
DEFAULT_BASE_KEYWORD = 'love'
GEO = 'EN'
HL = 'en-US'
CURRENT_YEAR = datetime.now().year
FORECAST_YEAR = CURRENT_YEAR
START_YEAR = CURRENT_YEAR - 7
END_YEAR = CURRENT_YEAR - 1

AUTHOR_NAME = "Eduard Samokhvalov"
AUTHOR_TITLE = "Quantitative Developer"
CONTACT_INFO = "edward.samokhvalov@gmail.com | Telegram: @EduardSam"

MIN_SLEEP_BETWEEN_KEYWORDS = 40
MAX_SLEEP_BETWEEN_KEYWORDS = 60
MIN_SLEEP_BETWEEN_API_CALLS = 5
MAX_SLEEP_BETWEEN_API_CALLS = 10

DATA_DIR_BASE = "data"

# ==================== HELPERS ====================
def create_retry_session(retries=5, backoff_factor=2.0):
    session = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries,
                  backoff_factor=backoff_factor,
                  allowed_methods=["GET", "POST"],
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def expand_keywords(base_word, max_suggestions=3):
    expanded = {base_word}
    session = create_retry_session()
    print(f"  Expanding '{base_word}' via DataMuse...")
    try:
        resp = session.get(f'https://api.datamuse.com/words?ml={base_word}&max=10', timeout=15)
        if resp.status_code == 200:
            for item in resp.json():
                expanded.add(item['word'])
    except Exception as e:
        print(f"    DataMuse error: {e}")
    time.sleep(random.uniform(MIN_SLEEP_BETWEEN_API_CALLS, MAX_SLEEP_BETWEEN_API_CALLS))
    print(f"  Expanding '{base_word}' via Google Trends suggestions...")
    try:
        pytrends = TrendReq(hl=HL, tz=360, timeout=30, requests_args={'verify': True})
        suggestions = pytrends.suggestions(base_word)
        for s in suggestions[:max_suggestions]:
            if 'title' in s:
                expanded.add(s['title'])
    except Exception as e:
        print(f"    Google Suggestions error: {e}")
    return list(expanded)

def get_trends_data(keyword, base_keyword, start_year, end_year, geo, lang):
    keyword_dir = os.path.join(DATA_DIR_BASE, base_keyword)
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
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    timeframe = f"{start_date} {end_date}"
    print(f"    Downloading data for '{keyword}' ({start_year}-{end_year})...")
    try:
        pytrends = TrendReq(hl=lang, tz=360, timeout=45, requests_args={'verify': True})
        pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time()
        if df.empty or keyword not in df.columns:
            return None
        df_out = df.reset_index()[['date', keyword]].rename(columns={'date': 'ds', keyword: 'y'})
        df_out['ds'] = pd.to_datetime(df_out['ds'])
        df_out = df_out.sort_values('ds')
        df_out.to_csv(filename, index=False)
        return df_out
    except Exception as e:
        print(f"    Download failed: {e}")
        return None

def get_yearly_seasonality(df, keyword, forecast_year):
    """Return ONLY the normalized yearly component (additive) for the forecast year."""
    train = df[df['ds'].dt.year < forecast_year].copy()
    if train.empty:
        return None, None, []
    model = Prophet(seasonality_mode='additive', yearly_seasonality=True)
    model.fit(train)
    future_dates = pd.date_range(start=f"{forecast_year}-01-01", end=f"{forecast_year}-12-31", freq='D')
    future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)
    yearly = forecast['yearly'].values          # <-- only yearly component
    # Normalize to range [-1, 1]
    if np.max(np.abs(yearly)) > 0:
        norm_yearly = yearly / np.max(np.abs(yearly))
    else:
        norm_yearly = yearly
    peaks, _ = find_peaks(norm_yearly, prominence=0.1)
    peak_dates = [future_dates[i] for i in peaks]
    return future_dates, norm_yearly, peak_dates

def load_existing_data(base_keyword):
    keyword_dir = os.path.join(DATA_DIR_BASE, base_keyword)
    if not os.path.isdir(keyword_dir):
        print(f"ERROR: Directory {keyword_dir} does not exist.")
        sys.exit(1)
    csv_files = [f for f in os.listdir(keyword_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"ERROR: No CSV files in {keyword_dir}.")
        sys.exit(1)
    data = {}
    for f in csv_files:
        kw = os.path.splitext(f)[0].replace('_', ' ')
        filepath = os.path.join(keyword_dir, f)
        try:
            df = pd.read_csv(filepath, parse_dates=['ds'])
            if 'ds' in df.columns and 'y' in df.columns:
                data[kw] = df
                print(f"  Loaded {kw}")
            else:
                print(f"  Skipping {filepath}: missing 'ds' or 'y'")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
    return data

def create_weekly_heatmap(all_curves, all_dates, keywords):
    df_all = pd.DataFrame(index=all_dates)
    for i, kw in enumerate(keywords):
        df_all[kw] = all_curves[i]
    df_all['avg'] = df_all.mean(axis=1)
    df_all['week_start'] = df_all.index - pd.to_timedelta(df_all.index.dayofweek, unit='d')
    df_all['week_start_date'] = df_all['week_start'].dt.date
    df_all['day_name'] = df_all.index.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_all['week_end'] = df_all['week_start'] + pd.Timedelta(days=6)
    df_all['week_label'] = df_all['week_start'].dt.strftime('%b %d') + ' – ' + df_all['week_end'].dt.strftime('%b %d')
    pivot = df_all.pivot_table(index='week_start_date', columns='day_name', values='avg', aggfunc='mean')
    pivot = pivot[day_order]
    week_labels = df_all.groupby('week_start_date')['week_label'].first().reindex(pivot.index)
    first_monday = df_all['week_start'].min().date()
    last_sunday = df_all['week_start'].max().date() + timedelta(days=6)
    all_weeks = pd.date_range(first_monday, last_sunday, freq='W-MON').date
    pivot = pivot.reindex(all_weeks).fillna(0)
    week_labels = week_labels.reindex(all_weeks).fillna('')
    fig = px.imshow(pivot,
                    labels=dict(x="Day of week", y="Week (Monday – Sunday)", color="Avg influence"),
                    title="Weekly Activity Heatmap (average normalized yearly influence)",
                    color_continuous_scale=['white', 'red'],
                    aspect='auto')
    fig.update_layout(height=600, width=None, autosize=True, margin=dict(l=140, r=20, t=60, b=40))
    fig.update_yaxes(tickvals=list(range(len(week_labels))), ticktext=week_labels.tolist(), tickangle=0)
    fig.update_traces(hovertemplate='Week: %{y}<br>Day: %{x}<br>Influence: %{z:.3f}<extra></extra>')
    return fig

def find_optimal_clusters(matrix, max_clusters=6):
    if matrix.shape[0] < 2:
        return 1
    best_n = 2
    best_score = -1
    for n in range(2, min(max_clusters, matrix.shape[0]) + 1):
        if n >= matrix.shape[0]:
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
    parser.add_argument('--data-ready', action='store_true')
    parser.add_argument('--base-keyword', type=str, default=DEFAULT_BASE_KEYWORD)
    args = parser.parse_args()
    base_keyword = args.base_keyword
    data_ready = args.data_ready

    print("=== Yearly Seasonality Only (Additive) ===")
    print(f"Base keyword: {base_keyword}")
    print(f"Data: {START_YEAR}-{END_YEAR} | Forecast: {FORECAST_YEAR}\n")

    if data_ready:
        keywords_data = load_existing_data(base_keyword)
    else:
        semantic_core = expand_keywords(base_keyword)
        keywords_data = {}
        for kw in semantic_core:
            time.sleep(random.uniform(MIN_SLEEP_BETWEEN_KEYWORDS, MAX_SLEEP_BETWEEN_KEYWORDS))
            df = get_trends_data(kw, base_keyword, START_YEAR, END_YEAR, GEO, HL)
            if df is not None:
                keywords_data[kw] = df

    if not keywords_data:
        print("No data. Exiting.")
        return

    # Extract yearly seasonality for each keyword
    yearly_data = {}
    all_peaks = []
    all_curves = []
    for kw, df in keywords_data.items():
        print(f"  {kw}...")
        dates, curve, peaks = get_yearly_seasonality(df, kw, FORECAST_YEAR)
        if dates is not None:
            yearly_data[kw] = (dates, curve, peaks)
            all_curves.append(curve)
            for p in peaks:
                all_peaks.append({'date': p, 'keyword': kw})

    if not yearly_data:
        print("No yearly seasonality extracted.")
        return

    # Clustering on the yearly curves
    first_dates = next(iter(yearly_data.values()))[0]
    n_days = len(first_dates)
    matrix = []
    for kw, (_, curve, _) in yearly_data.items():
        if len(curve) != n_days:
            curve = np.interp(np.linspace(0,1,n_days), np.linspace(0,1,len(curve)), curve)
        matrix.append(curve)
    matrix = np.array(matrix)
    n_clusters = find_optimal_clusters(matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix)
    keyword_list = list(yearly_data.keys())

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<title>Yearly Seasonality – {base_keyword.upper()} {FORECAST_YEAR}</title>
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
<body>
<div class="container">
<h1>Yearly Seasonal Patterns – {base_keyword.upper()} ({FORECAST_YEAR})</h1>
<p>Additive seasonality, normalized. Triangles = peaks.</p>
<div class="annotation">
    <strong>Only yearly component</strong> – no trend, no holidays.<br>
    Hover for exact date and influence. Zoom with mouse wheel.<br>
    Clusters computed via silhouette score.
</div>
"""
    # Cluster plots
    for c in range(n_clusters):
        cluster_kws = [keyword_list[i] for i,lbl in enumerate(labels) if lbl==c]
        fig = go.Figure()
        for kw in cluster_kws:
            dates, curve, peaks = yearly_data[kw]
            fig.add_trace(go.Scatter(x=dates, y=curve, mode='lines', name=kw,
                                     hovertemplate='%{x|%Y-%m-%d}<br>Influence: %{y:.3f}<extra></extra>'))
            if peaks:
                peak_vals = [curve[list(dates).index(p)] for p in peaks]
                fig.add_trace(go.Scatter(x=peaks, y=peak_vals, mode='markers',
                                         marker=dict(symbol='triangle-up', size=10, color='red'),
                                         name=f'{kw} peaks', showlegend=False,
                                         hovertemplate='Peak: %{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>'))
        fig.update_layout(title=f'Cluster {c+1} – Yearly Seasonality', height=500, hovermode='closest')
        html += f'<div class="plot"><h3>Cluster {c+1}</h3><div id="cluster{c}"></div></div>'
        html += f'<script>Plotly.newPlot("cluster{c}", {fig.to_json()});</script>'

    # Weekly heatmap
    hm_fig = create_weekly_heatmap(all_curves, first_dates, keyword_list)
    html += f'<div class="plot"><h3>Weekly Heatmap (Average Influence)</h3><div id="heatmap"></div></div>'
    html += f'<script>Plotly.newPlot("heatmap", {hm_fig.to_json()});</script>'

    # Keyword vs Month peak table
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    peak_mat = pd.DataFrame('', index=keyword_list, columns=month_names)
    for kw, (_, _, peaks) in yearly_data.items():
        for p in peaks:
            m = month_names[p.month-1]
            d = p.strftime('%d/%m')
            cur = peak_mat.loc[kw, m]
            peak_mat.loc[kw, m] = f"{cur}, {d}" if cur else d
    html += f'<div class="plot">{peak_mat.to_html(escape=False)}</div>'

    # All peaks table
    if all_peaks:
        pdf = pd.DataFrame(all_peaks).sort_values('date')
        pdf['date'] = pdf['date'].dt.strftime('%Y-%m-%d')
        html += f'<div class="plot">{pdf.to_html(index=False)}</div>'

    html += f'<div class="footer">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>Prophet additive yearly seasonality | Clusters: {n_clusters}</div>'
    html += '</div></body></html>'

    out = f"./Strategic_Trend_Report_{base_keyword.replace(' ','_')}_{FORECAST_YEAR}.html"
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✅ Report saved: {out}")
    print(f"   Keywords: {len(yearly_data)} | Clusters: {n_clusters} | Peaks: {len(all_peaks)}")

if __name__ == "__main__":
    main()
