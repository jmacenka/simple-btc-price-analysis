# Updated Dash app: adds a "Refresh Price-Data" button that re-fetches yfinance data
# and updates all charts, tables, and date pickers accordingly. Also uses a dcc.Store
# to hold the latest dataset.

import dash
from dash import dcc, html, Input, Output, dash_table, no_update
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np

# --------------------------
# Helper Functions
# --------------------------

def compute_rsi(series, window=14):
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def format_duration_in_days(days):
    """Convert a number of days to the best unit: days, weeks, months, or years."""
    if days < 7:
        return f"{days} days"
    elif days < 30:
        weeks = days / 7
        return f"{round(weeks)} weeks"
    elif days < 365:
        months = days / 30
        return f"{round(months)} months"
    else:
        years = days / 365
        return f"{round(years,1)} years"

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance into 'Field_Ticker'."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c0 if (isinstance(c1, str) and c1 == '') else f"{c0}_{c1}"
            for (c0, c1) in df.columns.values
        ]
    return df

def get_price_column(price_field: str, ticker: str, columns: pd.Index) -> str:
    """
    Return a column name that matches the requested price_field and ticker across
    the various shapes yfinance may return.
    """
    candidates = [
        f"{price_field}_{ticker}",     # group_by='column' flattened
        f"{ticker}_{price_field}",     # group_by='ticker' flattened
        price_field,                   # flat without ticker
    ]
    if price_field == "Close":
        candidates.insert(1, f"Adj Close_{ticker}")
        candidates.append("Adj Close")
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(
        f"Price column for '{price_field}' & '{ticker}' not found. Available: {list(columns)}"
    )

# --------------------------
# Load Raw Data (BTC-EUR + BTC-USD, robust)
# --------------------------

TICKERS = ["BTC-EUR", "BTC-USD"]

def load_history():
    df = yf.download(
        TICKERS,
        start="2014-01-01",
        progress=False,
        group_by="column",   # (Field, Ticker) -> 'Field_Ticker' after flatten
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data for BTC-EUR / BTC-USD")
    df = df.reset_index()
    df = _flatten_columns(df)
    return df

def df_from_store_json(data_json: str) -> pd.DataFrame:
    """Rebuild a DataFrame from dcc.Store JSON (orient='split')."""
    try:
        df = pd.read_json(data_json, orient='split')
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception:
        # Fallback to a minimal non-empty DataFrame to keep UI alive
        return pd.DataFrame({
            "Date": pd.to_datetime(["2014-01-01"]),
            "Open": [0.0], "High": [0.0], "Low": [0.0], "Close": [0.0], "Adj Close": [0.0], "Volume": [0]
        })

try:
    df_raw = load_history()
except Exception:
    # Minimal fallback to keep the layout from crashing
    df_raw = pd.DataFrame({
        "Date": pd.to_datetime(["2014-01-01"]),
        "Open": [0.0], "High": [0.0], "Low": [0.0], "Close": [0.0], "Adj Close": [0.0], "Volume": [0]
    })

# --------------------------
# Default Trading Strategy Markdown
# --------------------------

default_strategy_md = """
**Trading Strategy:**  
- **Short-term:** *Price* < MA₍200d₎ and RSI < 30 ⇒ Buy; *Price* > MA₍200d₎ and RSI > 70 ⇒ Sell.  
- **Long-term:** *Price* < MA₍200w₎ and RSI < 30 ⇒ Buy.  
Bubble size ∝ |Price − MA|, max size = 10× line thickness.
"""

# --------------------------
# App Layout with Two Tabs + Refresh
# --------------------------

parameters_style = {
    'border': '1px solid #ccc',
    'padding': '20px',
    'margin': '20px',
    'borderRadius': '5px',
    'backgroundColor': '#f9f9f9'
}

WARNING_STATEMENT = "For educational purpose only, not to be considered financial advise! This application comes as is and may contain critical flaws. Always do your own research before taking ANY investment decision! No guaranties or warranties given, what so ever!"

app = dash.Dash(__name__)
app.title = "Historic BTC Analytics (educational purpose only)"
server = app.server

# Precompute initial dates from current df_raw
_initial_min_date = pd.to_datetime(df_raw['Date']).min()
_initial_max_date = pd.to_datetime(df_raw['Date']).max()

app.layout = html.Div([
    # Store for the latest dataset (populated at load and refreshed by button)
    dcc.Store(
        id='data-store',
        data=df_raw.to_json(date_format='iso', orient='split')
    ),

    html.H1("Historic bitcoin price analytics", style={'textAlign': 'center', 'marginTop': '20px'}),
    html.P(WARNING_STATEMENT, style={'textAlign': 'center', 'marginTop': '10px'}),

    dcc.Tabs(id="tabs", value="analytics", children=[
        dcc.Tab(label="ANALYTICS", value="analytics", children=[
            html.Div([
                # Top row: Axis Scale, Pair, Date Range Picker.
                html.Div([
                    html.Div([
                        html.Label("Axis Scale:"),
                        dcc.RadioItems(
                            id='axis-scale',
                            options=[
                                {'label': 'Standard', 'value': 'standard'},
                                {'label': 'Log Scale (Y)', 'value': 'log'},
                                {'label': 'Log-Log', 'value': 'loglog'},
                            ],
                            value='standard',
                            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                        )
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([
                        html.Label("Pair:"),
                        dcc.Dropdown(
                            id='pair',
                            options=[
                                {'label': 'BTC/EUR', 'value': 'BTC-EUR'},
                                {'label': 'BTC/USD', 'value': 'BTC-USD'},
                            ],
                            value='BTC-EUR',  # Default to EUR pair
                            clearable=False
                        )
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([
                        dcc.DatePickerRange(
                            id='date-range-picker',
                            min_date_allowed=_initial_min_date,
                            max_date_allowed=_initial_max_date,
                            start_date=_initial_min_date,
                            end_date=_initial_max_date,
                            display_format='YYYY-MM-DD'
                        )
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'right'})
                ], style={'padding': '10px'}),

                dcc.Graph(id='combined-chart'),

                dcc.Markdown(id='strategy-markdown', children=default_strategy_md,
                             style={'textAlign': 'center', 'margin': '20px', 'fontSize': '14px'}),

                html.H2(id="quarterly-title", style={'textAlign': 'center', 'marginTop': '20px'}),
                dash_table.DataTable(
                    id='predicted-table',
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center'},
                ),

                html.H2("Debug Information", style={'textAlign': 'center', 'marginTop': '20px'}),
                html.Div(id='debug-container', style={'textAlign': 'center', 'color': 'blue'})
            ])
        ]),
        dcc.Tab(label="PARAMETERS", value="parameters", children=[
            # Refresh bar
            html.Div([
                html.Button("🔄 Refresh Price-Data", id='refresh-button', n_clicks=0, style={'marginRight': '10px'}),
                html.Span(
                    id='refresh-status',
                    children=f"Data through {_initial_max_date.date()} — click to fetch latest.",
                    style={'color': '#555'}
                )
            ], style={'textAlign': 'center', 'margin': '10px'}),
            html.Div([
                html.Label("Price Field:"),
                dcc.Dropdown(
                    id='price-field',
                    options=[
                        {'label': 'Close', 'value': 'Close'},
                        {'label': 'High', 'value': 'High'},
                        {'label': 'Low', 'value': 'Low'},
                        {'label': 'Open', 'value': 'Open'}
                    ],
                    value='Close',
                    clearable=False
                ),
                html.Br(),
                html.Label("Short-term MA Window (days):"),
                # Default short-term MA set to 180 days (~6 months).
                dcc.Input(id='short-term-ma-window', type='number', value=180),
                html.Br(), html.Br(),
                html.Label("Long-term MA Window (weeks):"),
                # Default long-term MA set to 208 weeks (~4 years).
                dcc.Input(id='long-term-ma-window', type='number', value=208),
                html.Br(), html.Br(),
                html.Label("RSI Overbought Threshold:"),
                dcc.Input(id='rsi-overbought', type='number', value=70),
                html.Br(), html.Br(),
                html.Label("RSI Oversold Threshold:"),
                dcc.Input(id='rsi-oversold', type='number', value=30),
                html.Br(), html.Br(),
                html.Label("Marker Scale Factor:"),
                dcc.Input(id='marker-scale-factor', type='number', value=1),
                html.Br(), html.Br(),
                html.Label("Historic Price Line Thickness:"),
                dcc.Input(id='price-line-thickness', type='number', value=2),
                html.Br(), html.Br(),
            ], style=parameters_style)
        ])
    ]),
    html.P(WARNING_STATEMENT, style={'textAlign': 'center', 'marginTop': '20px'})
])

# --------------------------
# Callback: Refresh Data (yfinance download)
# --------------------------

@app.callback(
    [Output('data-store', 'data'),
     Output('refresh-status', 'children')],
    Input('refresh-button', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_data(n_clicks):
    try:
        df_new = load_history()
        data_json = df_new.to_json(date_format='iso', orient='split')
        status = (f"Refreshed at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} — "
                  f"data through {pd.to_datetime(df_new['Date']).max().date()}.")
        return data_json, status
    except Exception as e:
        return no_update, f"Refresh failed: {e}"

# --------------------------
# Callback: Keep DatePicker in sync with (refreshed) data
# --------------------------

@app.callback(
    [Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed'),
     Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date')],
    Input('data-store', 'data')
)
def sync_date_picker(data_json):
    df = df_from_store_json(data_json)
    df['Date'] = pd.to_datetime(df['Date'])
    dmin = df['Date'].min()
    dmax = df['Date'].max()
    return dmin, dmax, dmin, dmax

# --------------------------
# Callback to Update Analytics
# --------------------------

@app.callback(
    [Output('combined-chart', 'figure'),
     Output('predicted-table', 'data'),
     Output('quarterly-title', 'children'),
     Output('debug-container', 'children'),
     Output('strategy-markdown', 'children')],
    [Input('axis-scale', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('pair', 'value'),
     Input('price-field', 'value'),
     Input('short-term-ma-window', 'value'),
     Input('long-term-ma-window', 'value'),
     Input('rsi-overbought', 'value'),
     Input('rsi-oversold', 'value'),
     Input('marker-scale-factor', 'value'),
     Input('price-line-thickness', 'value'),
     Input('data-store', 'data')]
)
def update_analytics(axis_scale, start_date, end_date, pair, price_field,
                     short_ma_window, long_ma_window, rsi_ob, rsi_os, marker_scale, price_line_thickness,
                     data_json):

    # Load latest data from store
    df_local = df_from_store_json(data_json).copy()
    df_local['Date'] = pd.to_datetime(df_local['Date'])
    df_local.sort_values('Date', inplace=True)
    df_local['Days'] = (df_local['Date'] - df_local['Date'].min()).dt.days + 1

    # Determine price column robustly.
    price_col = get_price_column(price_field, pair, df_local.columns)

    # Recalculate moving averages.
    short_ma_window = int(short_ma_window or 180)
    long_ma_window = int(long_ma_window or 208)

    df_local['MA_200d'] = df_local[price_col].rolling(window=short_ma_window).mean()
    df_week = df_local.set_index('Date').resample('W').last()
    df_week['MA_200w'] = df_week[price_col].rolling(window=long_ma_window).mean()
    ma200w_series = df_week['MA_200w'].reindex(df_local['Date'], method='ffill')
    df_local['MA_200w'] = ma200w_series.values

    # Recalculate RSI.
    df_local['RSI'] = compute_rsi(df_local[price_col])
    df_local['RSI_class'] = np.where(df_local['RSI'] > rsi_ob, 'overbought',
                              np.where(df_local['RSI'] < rsi_os, 'oversold', 'neutral'))

    # Define color dictionaries.
    price_colors = {'overbought': 'red', 'oversold': 'green', 'neutral': 'blue'}
    rsi_line_colors = {'overbought': 'red', 'oversold': 'green', 'neutral': 'black'}

    # Marker size (respect user's marker-scale-factor).
    max_marker_size = max(1, (price_line_thickness or 1)) * 10 * (marker_scale or 1)

    # Compute distances.
    df_local['ST_dist'] = (df_local[price_col] - df_local['MA_200d']).abs()
    df_local['LT_dist'] = (df_local[price_col] - df_local['MA_200w']).abs()

    def scale_marker_category(df_cat, dist_col, size_col):
        if not df_cat.empty:
            max_dist = df_cat[dist_col].max()
            if max_dist and max_dist > 0:
                df_cat[size_col] = (df_cat[dist_col] / max_dist) * max_marker_size
            else:
                df_cat[size_col] = max_marker_size / 2.0
        return df_cat

    short_term_buy = df_local[(df_local['RSI'] < rsi_os) & (df_local[price_col] < df_local['MA_200d'])].copy()
    short_term_sell = df_local[(df_local['RSI'] > rsi_ob) & (df_local[price_col] > df_local['MA_200d'])].copy()
    long_term_buy = df_local[(df_local['RSI'] < rsi_os) & (df_local[price_col] < df_local['MA_200w'])].copy()

    short_term_buy = scale_marker_category(short_term_buy, 'ST_dist', 'ST_marker_size')
    short_term_sell = scale_marker_category(short_term_sell, 'ST_dist', 'ST_marker_size')
    long_term_buy = scale_marker_category(long_term_buy, 'LT_dist', 'LT_marker_size')

    # Use full data for computing indicators.
    df_chart = df_local.copy()
    if axis_scale == 'loglog':
        df_chart = df_chart[df_chart['Days'] >= 365].copy()

    x_axis_col = 'Date'
    if axis_scale == 'loglog':
        x_axis_col = 'Days'

    base_date = df_local['Date'].min()
    if axis_scale == 'loglog':
        customdata_price = [(base_date + pd.Timedelta(days=int(x)-1)).strftime('%Y-%m-%d') for x in df_chart[x_axis_col]]
        customdata_rsi = customdata_price
    else:
        customdata_price = df_chart['Date'].dt.strftime('%Y-%m-%d')
        customdata_rsi = customdata_price

    # ------------------------------
    # Compute Power-Law Best-Fit for Predictions.
    # ------------------------------
    df_log = df_local[df_local['Days'] >= 365]
    valid = df_log[price_col] > 0
    x_fit_data = df_log.loc[valid, 'Days']
    y_fit_data = df_log.loc[valid, price_col]
    if len(x_fit_data) >= 2:
        log_x = np.log(x_fit_data)
        log_y = np.log(y_fit_data)
        slope_local, intercept_local = np.polyfit(log_x, log_y, 1)
        a_local = np.exp(intercept_local)
    else:
        slope_local, a_local = 0.0, float(y_fit_data.iloc[-1] if len(y_fit_data) else 1.0)

    # Show predictions for a symmetric range of quarters centered on the current quarter.
    current_quarter = pd.Timestamp.today().to_period('Q').start_time
    num_quarters_each_side = 4
    start_table = current_quarter - pd.DateOffset(months=3*num_quarters_each_side)
    end_table = current_quarter + pd.DateOffset(months=3*num_quarters_each_side)
    quarter_dates = pd.date_range(start=start_table, end=end_table, freq='QS')

    predicted_data = []
    last_hist_date = df_local['Date'].max()
    for q_date in quarter_dates:
        days_since_start = (q_date - df_local['Date'].min()).days + 1
        predicted_price = a_local * (days_since_start ** slope_local) if days_since_start > 0 else np.nan
        predicted_price_formatted = f"{predicted_price:,.0f}".replace(",", "'") if np.isfinite(predicted_price) else ""
        if q_date <= last_hist_date:
            historic_row = df_local[df_local['Date'] <= q_date]
            historic_price = historic_row.iloc[-1][price_col] if not historic_row.empty else None
        else:
            historic_price = None
        if (historic_price is not None) and np.isfinite(predicted_price) and predicted_price != 0:
            rel_div = 100 * (historic_price - predicted_price) / predicted_price
        else:
            rel_div = None
        quarter_label = f"{q_date.year} Q{q_date.quarter}"
        predicted_data.append({
            "Quarter": quarter_label,
            "Days Since Start": int(days_since_start),
            "Predicted Price": predicted_price_formatted,
            "Historic Price": f"{historic_price:,.0f}".replace(",", "'") if historic_price is not None else "",
            "Relative Divergence (%)": round(rel_div, 2) if rel_div is not None else ""
        })
    df_predictions_new = pd.DataFrame(predicted_data)

    # ------------------------------
    # Build Combined Chart (Price & RSI)
    # ------------------------------
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.7, 0.3],
                        subplot_titles=("Price Chart", "RSI"))

    # Base trace for Price using the neutral color.
    fig.add_trace(
        go.Scatter(
            x=df_chart[x_axis_col],
            y=df_chart[price_col],
            mode='lines',
            line=dict(color=price_colors['neutral'], width=price_line_thickness),
            showlegend=False,
            hovertemplate="Date: %{customdata}<br>Price: %{y:,.0f}<extra></extra>",
            customdata=customdata_price
        ),
        row=1, col=1
    )

    # Segmented Price trace.
    df_price = df_chart.sort_values('Date').copy()
    df_price['Group'] = (df_price['RSI_class'] != df_price['RSI_class'].shift()).cumsum()
    for _, group in df_price.groupby('Group'):
        x_vals = group[x_axis_col]
        customdata_group = ([(base_date + pd.Timedelta(days=int(x)-1)).strftime('%Y-%m-%d') for x in group[x_axis_col]]
                            if axis_scale=='loglog' else group['Date'].dt.strftime('%Y-%m-%d'))
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=group[price_col],
                mode='lines',
                line=dict(color=price_colors.get(group['RSI_class'].iloc[0], 'blue')),
                showlegend=False,
                hovertemplate="Date: %{customdata}<br>Price: %{y:,.0f}<extra></extra>",
                customdata=customdata_group
            ),
            row=1, col=1
        )

    x_vals_all = df_chart[x_axis_col]
    short_label = f"Short-Term {format_duration_in_days(short_ma_window)} MA"
    long_label = f"Long-Term {format_duration_in_days(long_ma_window * 7)} MA"

    fig.add_trace(
        go.Scatter(
            x=x_vals_all,
            y=df_chart['MA_200d'],
            mode='lines',
            line=dict(color='orange', width=price_line_thickness),
            name=short_label,
            hovertemplate="Date: %{customdata}<br>Price: %{y:,.0f}<extra></extra>",
            customdata=customdata_price
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals_all,
            y=df_chart['MA_200w'],
            mode='lines',
            line=dict(color='purple', width=price_line_thickness),
            name=long_label,
            hovertemplate="Date: %{customdata}<br>Price: %{y:,.0f}<extra></extra>",
            customdata=customdata_price
        ),
        row=1, col=1
    )

    # Overlay opportunity bubbles (long-term sell removed).
    bubble_traces = [
        (short_term_buy, 'Short-term Buy', 'lightgreen', 'circle'),
        (short_term_sell, 'Short-term Sell', 'lightcoral', 'circle'),
        (long_term_buy, 'Long-term Buy', 'green', 'square')
    ]
    for df_cat, name, color, symbol in bubble_traces:
        if not df_cat.empty:
            df_cat_chart = df_cat.copy()
            if axis_scale == 'loglog':
                df_cat_chart = df_cat_chart[df_cat_chart['Days'] >= 365]
                customdata_bubble = [(base_date + pd.Timedelta(days=int(x)-1)).strftime('%Y-%m-%d') for x in df_cat_chart[x_axis_col]]
            else:
                customdata_bubble = df_cat_chart['Date'].dt.strftime('%Y-%m-%d')
            # pick last created marker-size column
            size_col = [c for c in df_cat_chart.columns if c.endswith("_marker_size")]
            size_series = df_cat_chart[size_col[-1]] if size_col else pd.Series([6]*len(df_cat_chart), index=df_cat_chart.index)
            fig.add_trace(
                go.Scatter(
                    x=df_cat_chart[x_axis_col],
                    y=df_cat_chart[price_col],
                    mode='markers',
                    marker=dict(
                        size=size_series,
                        color=color,
                        opacity=1.0,
                        symbol=symbol
                    ),
                    name=name,
                    hovertemplate="Date: %{customdata}<br>Price: %{y:,.0f}<extra></extra>",
                    customdata=customdata_bubble
                ),
                row=1, col=1
            )

    # Overlay power-law fit line.
    if axis_scale == 'loglog':
        x_min = df_chart['Days'].min()
        x_max = df_chart['Days'].max()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = a_local * (x_line ** slope_local)
        mask = y_line >= 100
        x_line = x_line[mask]
        y_line = y_line[mask]
        customdata_fit = [(base_date + pd.Timedelta(days=int(x)-1)).strftime('%Y-%m-%d') for x in x_line]
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Power-Law Fit',
                hovertemplate="Date: %{customdata}<br>Price: %{y:,.0f}<extra></extra>",
                customdata=customdata_fit
            ),
            row=1, col=1
        )
    else:
        x_min = df_local['Days'].min()
        x_max = df_local['Days'].max()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = a_local * (x_line ** slope_local)
        mask = y_line >= 100
        x_line = x_line[mask]
        y_line = y_line[mask]
        date_min = df_local['Date'].min()
        x_line_dates = [date_min + pd.Timedelta(days=int(x)-1) for x in x_line]
        customdata_fit = [d.strftime('%Y-%m-%d') for d in x_line_dates]
        fig.add_trace(
            go.Scatter(
                x=x_line_dates,
                y=y_line,
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Power-Law Fit',
                hovertemplate="Date: %{customdata}<br>Price: %{y:,.0f}<extra></extra>",
                customdata=customdata_fit
            ),
            row=1, col=1
        )

    if axis_scale == 'loglog':
        fig.update_xaxes(type="log", row=1, col=1)
        days_min = df_chart['Days'].min()
        days_max = df_chart['Days'].max()
        tick_vals = np.logspace(np.log10(days_min if days_min > 0 else 1), np.log10(days_max), num=10)
        tick_vals = np.unique(np.round(tick_vals).astype(int))
        tick_text = [(df_chart['Date'].min() + pd.Timedelta(days=val-1)).strftime('%Y-%m-%d') for val in tick_vals]
        fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text, row=1, col=1)
    else:
        fig.update_xaxes(type="date", row=1, col=1)

    if axis_scale in ['log', 'loglog']:
        fig.update_yaxes(type="log", row=1, col=1, autorange=True)
    else:
        fig.update_yaxes(type="linear", row=1, col=1, autorange=True)

    # RSI Chart.
    fig.add_trace(
        go.Scatter(
            x=df_chart[x_axis_col],
            y=df_chart['RSI'],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hovertemplate="Date: %{customdata}<br>RSI: %{y:.0f}<extra></extra>",
            customdata=customdata_rsi
        ),
        row=2, col=1
    )
    df_rsi = df_chart.sort_values('Date').copy()
    df_rsi['Group'] = (df_rsi['RSI_class'] != df_rsi['RSI_class'].shift()).cumsum()
    for _, group in df_rsi.groupby('Group'):
        if axis_scale == 'loglog':
            customdata_group = [(base_date + pd.Timedelta(days=int(x)-1)).strftime('%Y-%m-%d') for x in group[x_axis_col]]
        else:
            customdata_group = group['Date'].dt.strftime('%Y-%m-%d')
        x_vals_rsi = group[x_axis_col]
        fig.add_trace(
            go.Scatter(
                x=x_vals_rsi,
                y=group['RSI'],
                mode='lines',
                line=dict(color=rsi_line_colors.get(group['RSI_class'].iloc[0], 'black')),
                showlegend=False,
                hovertemplate="Date: %{customdata}<br>RSI: %{y:.0f}<extra></extra>",
                customdata=customdata_group
            ),
            row=2, col=1
        )
    x_ref = df_chart['Date'].min() if axis_scale != 'loglog' else df_chart['Days'].min()
    x_max_val = df_chart['Date'].max() if axis_scale != 'loglog' else df_chart['Days'].max()
    # RSI zones
    fig.add_shape(
        type="rect",
        xref="x", yref="y2",
        x0=x_ref, x1=x_max_val,
        y0=70, y1=100,
        fillcolor="rgba(255,0,0,0.1)",
        line_width=0,
        layer="below",
        row=2, col=1
    )
    fig.add_shape(
        type="rect",
        xref="x", yref="y2",
        x0=x_ref, x1=x_max_val,
        y0=0, y1=30,
        fillcolor="rgba(0,255,0,0.1)",
        line_width=0,
        layer="below",
        row=2, col=1
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    if axis_scale == 'loglog':
        fig.update_xaxes(type="log", row=2, col=1)
        fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text, row=2, col=1)
    else:
        fig.update_xaxes(type="date", row=2, col=1)

    fig.update_layout(
        height=700,
        title_text=f"{pair} Price & RSI (Coupled X-axis)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Update x-axis range (zoom) and auto-adapt Price y-axis to the subset within the current date-range.
    if start_date and end_date:
        # Clip start/end dates to available range just in case
        s_date = pd.to_datetime(start_date)
        e_date = pd.to_datetime(end_date)
        data_min = df_local['Date'].min()
        data_max = df_local['Date'].max()
        s_date = max(s_date, data_min)
        e_date = min(e_date, data_max)

        if axis_scale == 'loglog':
            d0 = df_local['Date'].min()
            start_day = (s_date - d0).days + 1
            end_day = (e_date - d0).days + 1
            start_day = max(start_day, 1)
            end_day = max(end_day, start_day + 1)
            fig.update_xaxes(range=[np.log10(start_day), np.log10(end_day)], row=1, col=1)
            fig.update_xaxes(range=[np.log10(start_day), np.log10(end_day)], row=2, col=1)
            subset = df_local[(df_local['Days'] >= start_day) & (df_local['Days'] <= end_day)]
        else:
            fig.update_xaxes(range=[s_date, e_date], row=1, col=1)
            fig.update_xaxes(range=[s_date, e_date], row=2, col=1)
            subset = df_local[(df_local['Date'] >= s_date) & (df_local['Date'] <= e_date)]
        if not subset.empty:
            if axis_scale in ['log', 'loglog']:
                log_ymin = np.log10(subset[price_col].min())
                log_ymax = np.log10(subset[price_col].max())
                margin_log = (log_ymax - log_ymin) * 0.05
                new_y_range = [log_ymin - margin_log, log_ymax + margin_log]
                fig.update_yaxes(range=new_y_range, row=1, col=1)
            else:
                ymin = subset[price_col].min()
                ymax = subset[price_col].max()
                margin = (ymax - ymin) * 0.05
                new_y_range = [ymin - margin, ymax + margin]
                fig.update_yaxes(range=new_y_range, row=1, col=1)

    debug_text = (f"Pair: {pair} | Data from {df_local['Date'].min().date()} to {df_local['Date'].max().date()} | "
                  f"Price Field: {price_field}, Short-term MA: {short_ma_window}d, Long-term MA: {long_ma_window}w, "
                  f"RSI thresholds: oversold < {rsi_os}, overbought > {rsi_ob}")
    quarterly_title = f"Quarterly Price Predictions (Power-Law: y = {a_local:.2e} · (days)^({slope_local:.2e}))"

    dynamic_strategy = f"""
**Trading Strategy:**  
- **Short-term:** Buy if *{price_field}* < Short-Term MA and RSI < {rsi_os}%; Sell if *{price_field}* > Short-Term MA and RSI > {rsi_ob}%.  
- **Long-term:** Buy if *{price_field}* < Long-Term MA and RSI < {rsi_os}%.  
Short-Term MA = {format_duration_in_days(short_ma_window)}; Long-Term MA = {format_duration_in_days(long_ma_window * 7)}.  
Bubble size ∝ |Price − MA|, max size = 10× line thickness × marker-scale.
"""

    return fig, df_predictions_new.to_dict('records'), quarterly_title, debug_text, dynamic_strategy



# --------------------------
# Run the App
# --------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8050)
