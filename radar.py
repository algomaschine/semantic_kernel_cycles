#!/usr/bin/env python3
"""
Full‑screen radar charts (daily + weekly + monthly) for all archetypes (2026 forecast).
Uses multiplicative/additive seasonality per keyword based on stationarity.
Generates overlay and individual grid charts for each time grain.
Also exports monthly seasonality curves and peak values for easy insight generation.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
pd.plotting.register_matplotlib_converters()

FORECAST_YEAR = 2026
DATA_DIR = "trends-bulk"                         # folder with all bulk CSV files
KERNEL_FILE = "archetype_kernels.json"           # keyword mapping

COLORS = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
    '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
    '#000075', '#808080', '#ffffff', '#000000'
]


def is_stationary(series):
    """Return True if series is stationary (ADF p-value < 0.05)."""
    s = series.dropna()
    if len(s) < 12:
        return False
    try:
        result = adfuller(s, maxlag=int(np.sqrt(len(s))))
        return result[1] < 0.05
    except Exception:
        return False


def forecast_keyword(df, keyword):
    """Train Prophet on monthly keyword data; return daily dates + normalized yearly curve."""
    train = df[['ds', keyword]].copy()
    train.columns = ['ds', 'y']
    train = train[train['ds'].dt.year < FORECAST_YEAR].dropna()
    if train.shape[0] < 12:
        return None, None

    mode = 'multiplicative' if not is_stationary(train['y']) else 'additive'
    model = Prophet(seasonality_mode=mode, yearly_seasonality=True)
    model.fit(train)

    future_dates = pd.date_range(
        start=f'{FORECAST_YEAR}-01-01',
        end=f'{FORECAST_YEAR}-12-31',
        freq='D'
    )
    future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)
    yearly = forecast['yearly'].values

    min_y, max_y = yearly.min(), yearly.max()
    if max_y - min_y > 0:
        norm = (yearly - min_y) / (max_y - min_y)
    else:
        norm = yearly - yearly[0]
    return future_dates.values, norm


def average_archetype_curve(df, keyword_columns, archetype_name):
    """Average normalized daily yearly curves across all keywords of an archetype."""
    print(f"Processing {archetype_name} ({len(keyword_columns)} keywords)")
    curves = []
    common_dates = None
    for kw in keyword_columns:
        if kw not in df.columns:
            print(f"  Warning: '{kw}' not found – skipped")
            continue
        dates, norm = forecast_keyword(df, kw)
        if dates is not None:
            curves.append(norm)
            if common_dates is None:
                common_dates = dates
    if not curves:
        return None, None
    avg_curve = np.mean(curves, axis=0)
    return common_dates, avg_curve


def daily_to_weekly(dates, curve):
    """Aggregate daily normalized curve to weekly means (ISO weeks)."""
    df = pd.DataFrame({'ds': pd.to_datetime(dates), 'value': curve})
    df['week'] = df['ds'].dt.isocalendar().week.astype(int)
    weekly = df.groupby('week')['value'].mean()
    weeks = sorted(weekly.index.tolist())
    weekly_curve = weekly.loc[weeks].values
    return weeks, weekly_curve


def daily_to_monthly(dates, curve):
    """Aggregate daily normalized curve to monthly means."""
    df = pd.DataFrame({'ds': pd.to_datetime(dates), 'value': curve})
    df['month'] = df['ds'].dt.month
    monthly = df.groupby('month')['value'].mean()
    months = list(range(1, 13))
    monthly_curve = monthly.loc[months].values
    return months, monthly_curve


def create_overlay_fig(all_data, time_grain='daily'):
    """Polar overlay figure (all archetypes). time_grain in {'daily','weekly','monthly'}."""
    if time_grain == 'daily':
        dates = all_data[0][1]
        theta = [(d.timetuple().tm_yday - 1) * 360 / len(dates) for d in pd.DatetimeIndex(dates).to_pydatetime()]
        month_angles, month_labels = _get_month_ticks_daily(dates)
    elif time_grain == 'weekly':
        weeks = all_data[0][1]  # list of week numbers
        n_weeks = len(weeks)
        theta = [(i * 360 / n_weeks) for i in range(n_weeks)]
        month_angles, month_labels = _get_month_ticks_weekly(weeks, FORECAST_YEAR)
    else:  # monthly
        n_months = len(all_data[0][1])  # should be 12
        theta = [(i * 360 / n_months) for i in range(n_months)]
        month_angles = theta[:]  # direct month angles
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']

    fig = go.Figure()
    for idx, (arch, _, curve) in enumerate(all_data):
        fig.add_trace(go.Scatterpolar(
            r=curve,
            theta=theta,
            mode='lines',
            name=arch.capitalize(),
            line=dict(color=COLORS[idx % len(COLORS)], width=2),
            fill='toself',
            opacity=0.3,
            hovertemplate=f'{arch.capitalize()}<br>' +
                          (f'Day: %{{theta}}°' if time_grain == 'daily' else
                           f'Week: %{{theta}}°' if time_grain == 'weekly' else
                           f'Month: %{{theta}}°') +
                          f'<br>Influence: %{{r:.3f}}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text=f"Archetype Seasonality Overlay ({time_grain}) – {FORECAST_YEAR} Forecast",
            font=dict(size=20, color='#1a237e'),
            x=0.5, xanchor='center'
        ),
        polar=dict(
            angularaxis=dict(
                tickmode='array',
                tickvals=month_angles,
                ticktext=month_labels,
                direction='clockwise',
                rotation=90,
                tickfont=dict(size=14)
            ),
            radialaxis=dict(title='Normalised influence', range=[0, 1])
        ),
        legend=dict(
            orientation='v', yanchor='top', y=1, xanchor='left', x=1.02,
            bgcolor='rgba(255,255,255,0.9)', font=dict(size=12),
            bordercolor='black', borderwidth=1
        ),
        autosize=True,
        margin=dict(l=80, r=150, t=100, b=80)
    )
    return fig


def create_individual_fig(all_data, time_grain='daily'):
    """3x4 grid of polar charts, one per archetype."""
    if time_grain == 'daily':
        dates = all_data[0][1]
        theta = [(d.timetuple().tm_yday - 1) * 360 / len(dates) for d in pd.DatetimeIndex(dates).to_pydatetime()]
        month_angles, month_labels = _get_month_ticks_daily(dates)
    elif time_grain == 'weekly':
        weeks = all_data[0][1]
        n_weeks = len(weeks)
        theta = [(i * 360 / n_weeks) for i in range(n_weeks)]
        month_angles, month_labels = _get_month_ticks_weekly(weeks, FORECAST_YEAR)
    else:  # monthly
        n_months = len(all_data[0][1])
        theta = [(i * 360 / n_months) for i in range(n_months)]
        month_angles = theta[:]
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']

    n_archetypes = len(all_data)
    cols = 4
    rows = int(np.ceil(n_archetypes / cols))
    specs = [[{'type': 'polar'} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=[arch.capitalize() for arch, _, _ in all_data]
    )

    for i, (arch, _, curve) in enumerate(all_data):
        row = i // cols + 1
        col = i % cols + 1
        fig.add_trace(
            go.Scatterpolar(
                r=curve,
                theta=theta,
                mode='lines',
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                fill='toself', opacity=0.5, showlegend=False,
                name=arch.capitalize()
            ),
            row=row, col=col
        )
        fig.update_polars(
            dict(
                angularaxis=dict(
                    tickmode='array',
                    tickvals=month_angles,
                    ticktext=month_labels,
                    direction='clockwise', rotation=90,
                    tickfont=dict(size=10)
                ),
                radialaxis=dict(range=[0, 1], showticklabels=False)
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=dict(
            text=f"Individual Archetype Radars ({time_grain}) – {FORECAST_YEAR} Forecast",
            font=dict(size=20, color='#1a237e'),
            x=0.5, xanchor='center'
        ),
        height=900, width=1200,
        showlegend=False
    )
    return fig


def _get_month_ticks_daily(dates):
    """Return monthly mid-month angles and labels for daily polar chart."""
    n_days = len(dates)
    month_angles = [((i * 30) + 15) * 360 / n_days for i in range(12)]
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    return month_angles, month_labels


def _get_month_ticks_weekly(week_numbers, year):
    """Return month label angles for weekly polar chart (week containing the 15th)."""
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    month_angles = []
    for month in range(1, 13):
        mid_date = pd.Timestamp(f'{year}-{month:02d}-15')
        week_of_mid = mid_date.isocalendar().week
        if week_of_mid in week_numbers:
            idx = week_numbers.index(week_of_mid)
        else:
            # nearest week
            idx = min(range(len(week_numbers)), key=lambda i: abs(week_numbers[i]-week_of_mid))
        angle = idx * 360 / len(week_numbers)
        month_angles.append(angle)
    return month_angles, month_labels


def save_html(fig, filename, title):
    """Save figure to a full‑screen HTML file."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body, html {{ width:100%; height:100%; background:#f0f2f5; }}
        .container {{ width:100vw; height:100vh; overflow:auto; }}
        .plotly-graph-div {{ width:100%; height:100%; min-height:600px; }}
    </style>
</head>
<body>
    <div class="container">
        {fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})}
    </div>
</body>
</html>"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Saved: {filename}")


def save_individual_html(fig, filename, title):
    """Save grid chart to responsive HTML."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body, html {{ margin:0; padding:0; background:#f0f2f5; }}
        .container {{ max-width:1400px; margin:auto; padding:20px; }}
        .plotly-graph-div {{ width:100%; }}
    </style>
</head>
<body>
    <div class="container">
        {fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})}
    </div>
</body>
</html>"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Saved: {filename}")


def main():
    # Load archetype kernels
    with open(KERNEL_FILE, 'r', encoding='utf-8') as f:
        kernel_map = json.load(f)

    # Read and concatenate all CSV files
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Folder '{DATA_DIR}' not found.")
        return
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found.")
        return

    dfs = []
    for fname in csv_files:
        path = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(path, parse_dates=['Time'])
            dfs.append(df.set_index('Time'))
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    if not dfs:
        return

    combined = pd.concat(dfs, axis=1, join='inner')
    combined.reset_index(inplace=True)
    combined.rename(columns={'Time': 'ds'}, inplace=True)
    print(f"Combined data shape: {combined.shape}")

    # Process all archetypes -> daily curves
    all_data_daily = []
    for archetype, keywords in kernel_map.items():
        all_columns = [archetype] + keywords
        available = [c for c in all_columns if c in combined.columns]
        if not available:
            print(f"Skipping {archetype}: no columns found.")
            continue
        dates, curve = average_archetype_curve(combined, available, archetype)
        if dates is not None:
            all_data_daily.append((archetype, dates, curve))
        else:
            print(f"Skipping {archetype}: no valid forecast.")

    if not all_data_daily:
        print("No archetype data for radars.")
        return

    # ----- Daily reports -----
    overlay_fig_daily = create_overlay_fig(all_data_daily, 'daily')
    save_html(overlay_fig_daily,
              "radar_overlay_all_archetypes_fullscreen.html",
              f"Archetype Overlay Radar (daily) – {FORECAST_YEAR}")

    ind_fig_daily = create_individual_fig(all_data_daily, 'daily')
    save_individual_html(ind_fig_daily,
                         "radar_individual_archetypes.html",
                         f"Individual Archetype Radars (daily) – {FORECAST_YEAR}")

    # ----- Weekly reports -----
    common_dates = all_data_daily[0][1]
    week_df = pd.DataFrame({'ds': pd.to_datetime(common_dates),
                            'week': pd.DatetimeIndex(common_dates).isocalendar().week})
    unique_weeks = sorted(week_df['week'].unique().tolist())

    all_data_weekly = []
    for arch, dates, daily_curve in all_data_daily:
        _, weekly_curve = daily_to_weekly(dates, daily_curve)
        if len(weekly_curve) == len(unique_weeks):
            all_data_weekly.append((arch, unique_weeks, weekly_curve))
        else:
            print(f"  Week count mismatch for {arch} – skipped weekly.")

    if all_data_weekly:
        overlay_fig_weekly = create_overlay_fig(all_data_weekly, 'weekly')
        save_html(overlay_fig_weekly,
                  "radar_overlay_weekly.html",
                  f"Archetype Overlay Radar (weekly) – {FORECAST_YEAR}")

        ind_fig_weekly = create_individual_fig(all_data_weekly, 'weekly')
        save_individual_html(ind_fig_weekly,
                             "radar_individual_weekly.html",
                             f"Individual Archetype Radars (weekly) – {FORECAST_YEAR}")
    else:
        print("Weekly data could not be generated.")

    # ----- Monthly reports -----
    all_data_monthly = []
    for arch, dates, daily_curve in all_data_daily:
        months, monthly_curve = daily_to_monthly(dates, daily_curve)
        # months list is always [1..12]
        all_data_monthly.append((arch, months, monthly_curve))

    if all_data_monthly:
        overlay_fig_monthly = create_overlay_fig(all_data_monthly, 'monthly')
        save_html(overlay_fig_monthly,
                  "radar_overlay_monthly.html",
                  f"Archetype Overlay Radar (monthly) – {FORECAST_YEAR}")

        ind_fig_monthly = create_individual_fig(all_data_monthly, 'monthly')
        save_individual_html(ind_fig_monthly,
                             "radar_individual_monthly.html",
                             f"Individual Archetype Radars (monthly) – {FORECAST_YEAR}")

        # ---------- Save monthly curves and peaks for analysis ----------
        # 1. Full monthly curves CSV
        months_list = all_data_monthly[0][1]  # [1,2,...,12]
        curves_dict = {'archetype': [arch for arch, _, _ in all_data_monthly]}
        for i, m in enumerate(months_list):
            curves_dict[f'month_{m:02d}'] = [curve[i] for _, _, curve in all_data_monthly]
        curves_df = pd.DataFrame(curves_dict)
        curves_df.to_csv('monthly_seasonality_curves.csv', index=False)
        print("✅ Saved: monthly_seasonality_curves.csv")

        # 2. Peaks CSV (max influence month + value, trough for context)
        peak_data = []
        for arch, mons, curve in all_data_monthly:
            idx_max = int(np.argmax(curve))
            idx_min = int(np.argmin(curve))
            peak_data.append({
                'archetype': arch,
                'peak_month': mons[idx_max],
                'peak_value': round(float(curve[idx_max]), 4),
                'trough_month': mons[idx_min],
                'trough_value': round(float(curve[idx_min]), 4)
            })
        peak_df = pd.DataFrame(peak_data)
        peak_df.to_csv('monthly_peaks.csv', index=False)
        print("✅ Saved: monthly_peaks.csv")
    else:
        print("Monthly data not available.")

    print("\nAll done! Generated 6 radar HTML files + 2 CSV data exports.")


if __name__ == "__main__":
    main()
