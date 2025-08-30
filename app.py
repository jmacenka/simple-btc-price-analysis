# Dash BTC Analytics App
# - Tabs: About (first), Analytics (default open), Parameters
# - Refresh price data button (yfinance)
# - dcc.Store holds latest dataset
# - Default RSI thresholds: Oversold=20, Overbought=80
# - RSI zones reflect dynamic thresholds
# - Price chart overlays buy/sell/long-term-buy signals
#   * Buy signals show Return% & ARR% vs latest (since signal date)
#   * Sell signals DO NOT show Return/ARR
# - Default view spans Long-term MA window (not full history)
#
# Updates in this version:
# - Added "About" tab with explanation + resource links (educational-only emphasis)
# - Human-readable power-law title (both plain & compact scientific for clarity)
# - Kept prior fixes (robust hover date formatting; humanized durations)

import io
import dash
from dash import dcc, html, Input, Output, dash_table, no_update
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np

# =========================
# Constants (DRY)
# =========================
TICKERS = ["BTC-EUR", "BTC-USD"]
DEFAULT_SHORT_TERM_DAYS = 180          # ~6 months
DEFAULT_LONG_TERM_WEEKS = 208          # ~4 years
DEFAULT_RSI_OVERBOUGHT = 80            # %
DEFAULT_RSI_OVERSOLD = 20              # %
DAYS_PER_YEAR = 365
DAYS_PER_MONTH = 30
YFIT_MIN_PRICE = 100                   # hide nonsensical low fitted values

PRICE_COLORS = {'overbought': 'red', 'oversold': 'green', 'neutral': 'blue'}
RSI_LINE_COLORS = {'overbought': 'red', 'oversold': 'green', 'neutral': 'black'}

HOVER_PRICE = "Date: %{customdata}<br>Price: %{y:,.0f}<extra></extra>"
HOVER_BUY   = ("Date: %{customdata[0]}<br>"
               "Price: %{y:,.0f}<br>"
               "Held: %{customdata[3]}<br>"
               "Return: %{customdata[1]:.1f}%<br>"
               "ARR: %{customdata[2]:.1f}%<extra></extra>")
HOVER_SELL  = ("Date: %{customdata}<br>"
               "Price: %{y:,.0f}<extra></extra>")
HOVER_RSI   = "Date: %{customdata}<br>RSI: %{y:.0f}<extra></extra>"

WARNING_STATEMENT = (
    "For educational purpose only, not to be considered financial advise! "
    "This application comes as is and may contain critical flaws. Always do your own research "
    "before taking ANY investment decision! No guaranties or warranties given, what so ever!"
)

PARAMETERS_STYLE = {
    'border': '1px solid #ccc',
    'padding': '20px',
    'margin': '20px',
    'borderRadius': '5px',
    'backgroundColor': '#f9f9f9'
}

# =========================
# Helpers
# =========================
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fmt_duration_days(days: int) -> str:
    if days < 7:
        return f"{days} days"
    if days < 30:
        return f"{round(days/7)} weeks"
    if days < DAYS_PER_YEAR:
        return f"{round(days/30)} months"
    return f"{round(days/DAYS_PER_YEAR,1)} years"

def humanize_days(days: int) -> str:
    """Convert integer days to 'Yy Mm Dd' (approx months=30d, years=365d)."""
    days = int(max(0, days))
    years = days // DAYS_PER_YEAR
    rem = days % DAYS_PER_YEAR
    months = rem // DAYS_PER_MONTH
    d = rem % DAYS_PER_MONTH
    parts = []
    if years: parts.append(f"{years}y")
    if months: parts.append(f"{months}m")
    parts.append(f"{d}d")
    return " ".join(parts)

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c0 if (isinstance(c1, str) and c1 == '') else f"{c0}_{c1}"
            for (c0, c1) in df.columns.values
        ]
    return df

def get_price_column(price_field: str, ticker: str, columns: pd.Index) -> str:
    candidates = [
        f"{price_field}_{ticker}",
        f"{ticker}_{price_field}",
        price_field,
    ]
    if price_field == "Close":
        candidates.insert(1, f"Adj Close_{ticker}")
        candidates.append("Adj Close")
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"Price column for '{price_field}' & '{ticker}' not found. Available: {list(columns)}")

def load_history() -> pd.DataFrame:
    df = yf.download(
        TICKERS,
        start="2014-01-01",
        progress=False,
        group_by="column",
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data for BTC-EUR / BTC-USD")
    return _flatten_columns(df.reset_index())

def df_from_store_json(data_json: str) -> pd.DataFrame:
    try:
        # Use StringIO to avoid FutureWarning for literal JSON strings
        df = pd.read_json(io.StringIO(data_json), orient='split')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception:
        return pd.DataFrame({
            "Date": pd.to_datetime(["2014-01-01"]),
            "Open": [0.0], "High": [0.0], "Low": [0.0], "Close": [0.0], "Adj Close": [0.0], "Volume": [0]
        })

def power_law_fit(days_series: pd.Series, price_series: pd.Series):
    """Return (a, slope) for y = a * x^slope using log-log OLS; fallback sane defaults."""
    valid = (days_series >= 365) & (price_series > 0)
    x = days_series[valid]
    y = price_series[valid]
    if len(x) >= 2:
        log_x = np.log(x)
        log_y = np.log(y)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        return float(np.exp(intercept)), float(slope)
    last = float(y.iloc[-1]) if len(y) else 1.0
    return last, 0.0

def add_perf_vs_latest(df_sig: pd.DataFrame, latest_date: pd.Timestamp, latest_price: float, price_col: str):
    """Add ReturnPct/ARRPct/DaysHeldHuman for BUY signals only."""
    if df_sig.empty:
        df_sig['ReturnPct'] = []
        df_sig['ARRPct'] = []
        df_sig['DaysHeldHuman'] = []
        return df_sig
    days = (latest_date - df_sig['Date']).dt.days.clip(lower=1)  # avoid /0
    growth = (latest_price / df_sig[price_col]).replace(0, np.nan)
    df_sig['ReturnPct'] = ((growth - 1.0) * 100.0).round(2)
    df_sig['ARRPct'] = ((growth ** (365.0 / days)) - 1.0).mul(100.0).round(2)
    df_sig['DaysHeldHuman'] = days.astype(int).apply(humanize_days)
    return df_sig

def scale_marker(df_cat: pd.DataFrame, dist_col: str, out_col: str, max_size: float) -> pd.DataFrame:
    if df_cat.empty:
        return df_cat
    md = df_cat[dist_col].max()
    df_cat[out_col] = (df_cat[dist_col] / md * max_size) if (md and md > 0) else (max_size/2.0)
    return df_cat

def marker_sizes(df_cat: pd.DataFrame) -> pd.Series:
    cols = [c for c in df_cat.columns if c.endswith("_marker_size")]
    return df_cat[cols[-1]] if cols else pd.Series([6]*len(df_cat), index=df_cat.index)

def dates_for_hover(x_vals, use_loglog: bool, base_date: pd.Timestamp):
    """
    Return a list of 'YYYY-MM-DD' strings for hover.
    - If use_loglog: x_vals are day numbers (array-like)
    - Else: x_vals are datetimes (Series, DatetimeIndex, ndarray)
    """
    if use_loglog:
        return [(base_date + pd.Timedelta(days=int(x) - 1)).strftime('%Y-%m-%d') for x in np.asarray(x_vals)]

    x = pd.to_datetime(x_vals)
    if isinstance(x, pd.Series):
        return x.dt.strftime('%Y-%m-%d').tolist()
    if isinstance(x, pd.DatetimeIndex):
        return x.strftime('%Y-%m-%d').tolist()
    return pd.Series(x).dt.strftime('%Y-%m-%d').tolist()

def fmt_power_law_readable(a: float, k: float) -> str:
    """
    Format 'y = a · days^k' in a human-friendly way:
    - shows a with up to 6 significant digits in plain decimal when reasonable,
      plus compact scientific in parentheses for clarity.
    - k with 2 decimal places.
    """
    # Plain decimal for 'a'
    if a == 0 or (abs(a) < 1e-6 or abs(a) >= 1e6):
        a_plain = f"{a:.6g}"
    else:
        a_plain = f"{a:.6f}".rstrip('0').rstrip('.')
    # Compact sci (a × 10^n)
    if a == 0:
        a_sci = "0"
    else:
        exp = int(np.floor(np.log10(abs(a))))
        mant = a / (10 ** exp)
        a_sci = f"{mant:.3f}×10^{exp}"
    return f"y = {a_plain} · (days)^{k:.2f}  ({a_sci})"

# =========================
# Data bootstrap
# =========================
try:
    df_raw = load_history()
except Exception:
    df_raw = pd.DataFrame({
        "Date": pd.to_datetime(["2014-01-01"]),
        "Open": [0.0], "High": [0.0], "Low": [0.0], "Close": [0.0], "Adj Close": [0.0], "Volume": [0]
    })

_INITIAL_MIN_DATE = pd.to_datetime(df_raw['Date']).min()
_INITIAL_MAX_DATE = pd.to_datetime(df_raw['Date']).max()
_INITIAL_START_DATE = max(_INITIAL_MIN_DATE, _INITIAL_MAX_DATE - pd.to_timedelta(DEFAULT_LONG_TERM_WEEKS * 7, unit='D'))

# =========================
# UI: About markdown (with links)
# =========================
ABOUT_MD = f"""
# About this App

**Purpose (Educational Only):**  
This app visualizes historical Bitcoin prices and momentum using:
- **Price chart** with **short-term** (e.g., 200 *days*) and **long-term** (e.g., 200 *weeks*) moving averages  
- **RSI (Relative Strength Index)** to highlight *overbought*/*oversold* regimes  
- Optional **log** and **log–log** views  
- A simple **power-law** fit used to build a *quarterly* prediction table (illustrative only)

> {WARNING_STATEMENT}

## How to read the charts

- **Axis scale:**  
  - *Standard* shows prices on linear time/price axes.  
  - *Log* sets the **price axis** to logarithmic (helpful across multiple orders of magnitude).  
  - *Log–log* sets **both axes** to logarithmic; straight lines often indicate power-law relationships.

- **Moving Averages:**  
  - **Short-term MA** (default {DEFAULT_SHORT_TERM_DAYS}d) approximates ~{fmt_duration_days(DEFAULT_SHORT_TERM_DAYS)} trend.  
  - **Long-term MA** (default {DEFAULT_LONG_TERM_WEEKS}w) approximates ~{fmt_duration_days(DEFAULT_LONG_TERM_WEEKS*7)} trend.  
  Crossovers and distance from MAs help contextualize price moves.

- **RSI (momentum):**  
  - We color the RSI and price segments by regime:  
    *oversold* (RSI < **{DEFAULT_RSI_OVERSOLD}**), *neutral*, *overbought* (RSI > **{DEFAULT_RSI_OVERBOUGHT}**).  
  - Shaded areas in the RSI panel mark these regions.

- **Signals (illustrative):**  
  - **Short-term Buy**: RSI < oversold **and** Price < Short-term MA.  
  - **Short-term Sell**: RSI > overbought **and** Price > Short-term MA.  
  - **Long-term Buy**: RSI < oversold **and** Price < Long-term MA.  
  Buy markers show **Return** and **ARR** (annualized rate of return) from that date to the latest point.

- **Predictions table:**  
  - Uses a **power-law** fit of price vs. time (days since first data point) to compute an indicative level per quarter.  
  - Displays a *human-readable* formula header (coefficient and exponent).  
  - Divergence column compares historical price (if available) to the model on that date.  
  - **Note:** This is *not* a forecast—purely an educational visualization of a simple curve-fit.

## Good background resources

- **RSI:**  
  - Investopedia – *Relative Strength Index (RSI)*: <https://www.investopedia.com/terms/r/rsi.asp>  
  - Investopedia – *RSI buy/sell signals*: <https://www.investopedia.com/articles/active-trading/042114/overbought-or-oversold-use-relative-strength-index-find-out.asp>

- **Moving Averages:**  
  - Investopedia – *Simple Moving Average (SMA)*: <https://www.investopedia.com/terms/s/sma.asp>  
  - Investopedia – *Why the 200-day SMA?*: <https://www.investopedia.com/ask/answers/013015/why-200-simple-moving-average-sma-so-common-traders-and-analysts.asp>  
  - Investopedia – *Moving Average (MA)* overview: <https://www.investopedia.com/terms/m/movingaverage.asp>

- **Logarithmic & Log–Log:**  
  - Wikipedia – *Logarithmic scale*: <https://en.wikipedia.org/wiki/Logarithmic_scale>  
  - Wikipedia – *Log–log plot*: <https://en.wikipedia.org/wiki/Log%E2%80%93log_plot>

- **Power laws & Bitcoin context:**  
  - Wikipedia – *Power law* (general): <https://en.wikipedia.org/wiki/Power_law>  
  - Giovanni Santostasi – *Bitcoin Power Law Theory*: <https://giovannisantostasi.medium.com/the-bitcoin-power-law-theory-962dfaf99ee9>  
  - Fulgur Ventures – *Bitcoin Power Law Theory — Executive Summary*: <https://medium.com/%40fulgur.ventures/bitcoin-power-law-theory-executive-summary-report-837e6f00347e>

- **ARR / CAGR concept:**  
  - Investopedia – *Compound Annual Growth Rate (CAGR)*: <https://www.investopedia.com/terms/c/cagr.asp>

- **Data source library:**  
  - yfinance docs: <https://ranaroussi.github.io/yfinance/>

Again: **This app is for education only. It is not investment advice.**
"""

# =========================
# UI
# =========================
default_strategy_md = f"""
**Trading Strategy:**  
- **Short-term:** *Price* < MA₍200d₎ and RSI < {DEFAULT_RSI_OVERSOLD} ⇒ Buy; *Price* > MA₍200d₎ and RSI > {DEFAULT_RSI_OVERBOUGHT} ⇒ Sell.  
- **Long-term:** *Price* < MA₍200w₎ and RSI < {DEFAULT_RSI_OVERSOLD} ⇒ Buy.  
Bubble size ∝ |Price − MA|, max size = 10× line thickness.
"""

app = dash.Dash(__name__)
app.title = "Historic BTC Analytics (educational purpose only)"
server = app.server

# Tabs: About (first), Analytics, Parameters; default value="analytics"
app.layout = html.Div([
    dcc.Store(id='data-store', data=df_raw.to_json(date_format='iso', orient='split')),

    html.H1("Historic bitcoin price analytics", style={'textAlign': 'center', 'marginTop': '20px'}),
    html.P(WARNING_STATEMENT, style={'textAlign': 'center', 'marginTop': '10px'}),

    dcc.Tabs(id="tabs", value="analytics", children=[
        dcc.Tab(label="ABOUT", value="about", children=[
            html.Div([
                dcc.Markdown(ABOUT_MD, link_target="_blank", style={'maxWidth': '900px', 'margin': '0 auto', 'padding': '20px'})
            ])
        ]),
        dcc.Tab(label="ANALYTICS", value="analytics", children=[
            html.Div([
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
                            value='BTC-EUR',
                            clearable=False
                        )
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([
                        dcc.DatePickerRange(
                            id='date-range-picker',
                            min_date_allowed=_INITIAL_MIN_DATE,
                            max_date_allowed=_INITIAL_MAX_DATE,
                            start_date=_INITIAL_START_DATE,
                            end_date=_INITIAL_MAX_DATE,
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
            html.Div([
                html.Button("🔄 Refresh Price-Data", id='refresh-button', n_clicks=0, style={'marginRight': '10px'}),
                html.Span(
                    id='refresh-status',
                    children=f"Data through {_INITIAL_MAX_DATE.date()} — click to fetch latest.",
                    style={'color': '#555'}
                )
            ], style={'textAlign': 'center', 'margin': '10px'}),
            html.Div([
                html.Label("Price Field:"),
                dcc.Dropdown(
                    id='price-field',
                    options=[{'label': x, 'value': x} for x in ['Close', 'High', 'Low', 'Open']],
                    value='Close',
                    clearable=False
                ),
                html.Br(),
                html.Label("Short-term MA Window (days):"),
                dcc.Input(id='short-term-ma-window', type='number', value=DEFAULT_SHORT_TERM_DAYS),
                html.Br(), html.Br(),
                html.Label("Long-term MA Window (weeks):"),
                dcc.Input(id='long-term-ma-window', type='number', value=DEFAULT_LONG_TERM_WEEKS),
                html.Br(), html.Br(),
                html.Label("RSI Overbought Threshold:"),
                dcc.Input(id='rsi-overbought', type='number', value=DEFAULT_RSI_OVERBOUGHT),
                html.Br(), html.Br(),
                html.Label("RSI Oversold Threshold:"),
                dcc.Input(id='rsi-oversold', type='number', value=DEFAULT_RSI_OVERSOLD),
                html.Br(), html.Br(),
                html.Label("Marker Scale Factor:"),
                dcc.Input(id='marker-scale-factor', type='number', value=1),
                html.Br(), html.Br(),
                html.Label("Historic Price Line Thickness:"),
                dcc.Input(id='price-line-thickness', type='number', value=2),
                html.Br(), html.Br(),
            ], style=PARAMETERS_STYLE)
        ])
    ]),
    html.P(WARNING_STATEMENT, style={'textAlign': 'center', 'marginTop': '20px'})
])

# =========================
# Callbacks
# =========================
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

@app.callback(
    [Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed'),
     Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date')],
    [Input('data-store', 'data'),
     Input('long-term-ma-window', 'value')]
)
def sync_date_picker(data_json, long_term_weeks):
    df = df_from_store_json(data_json)
    df['Date'] = pd.to_datetime(df['Date'])
    dmin, dmax = df['Date'].min(), df['Date'].max()
    lt_weeks = int(long_term_weeks or DEFAULT_LONG_TERM_WEEKS)
    start = max(dmin, dmax - pd.to_timedelta(lt_weeks * 7, unit='D'))
    return dmin, dmax, start, dmax

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

    # ---- Data prep
    df = df_from_store_json(data_json).copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days + 1

    price_col = get_price_column(price_field, pair, df.columns)

    short_ma_window = int(short_ma_window or DEFAULT_SHORT_TERM_DAYS)
    long_ma_window = int(long_ma_window or DEFAULT_LONG_TERM_WEEKS)

    df['MA_200d'] = df[price_col].rolling(window=short_ma_window).mean()
    df_week = df.set_index('Date').resample('W').last()
    df_week['MA_200w'] = df_week[price_col].rolling(window=long_ma_window).mean()
    df['MA_200w'] = df_week['MA_200w'].reindex(df['Date'], method='ffill').values

    df['RSI'] = compute_rsi(df[price_col])
    df['RSI_class'] = np.where(df['RSI'] > rsi_ob, 'overbought',
                        np.where(df['RSI'] < rsi_os, 'oversold', 'neutral'))

    max_marker_size = max(1, (price_line_thickness or 1)) * 10 * (marker_scale or 1)

    df['ST_dist'] = (df[price_col] - df['MA_200d']).abs()
    df['LT_dist'] = (df[price_col] - df['MA_200w']).abs()

    st_buy = scale_marker(df[(df['RSI'] < rsi_os) & (df[price_col] < df['MA_200d'])].copy(),
                          'ST_dist', 'ST_marker_size', max_marker_size)
    st_sell = scale_marker(df[(df['RSI'] > rsi_ob) & (df[price_col] > df['MA_200d'])].copy(),
                           'ST_dist', 'ST_marker_size', max_marker_size)
    lt_buy = scale_marker(df[(df['RSI'] < rsi_os) & (df[price_col] < df['MA_200w'])].copy(),
                          'LT_dist', 'LT_marker_size', max_marker_size)

    # Add performance ONLY for buy signals
    latest_price = df[price_col].iloc[-1]
    latest_date = df['Date'].iloc[-1]
    st_buy = add_perf_vs_latest(st_buy, latest_date, latest_price, price_col)
    lt_buy = add_perf_vs_latest(lt_buy, latest_date, latest_price, price_col)

    # Choose x axis mode
    use_loglog = (axis_scale == 'loglog')
    df_chart = df[df['Days'] >= 365].copy() if use_loglog else df.copy()
    x_axis_col = 'Days' if use_loglog else 'Date'
    base_date = df['Date'].min()

    # Power-law fit
    a_fit, slope_fit = power_law_fit(df['Days'], df[price_col])

    # Quarterly prediction table (±4 quarters around current quarter)
    current_q = pd.Timestamp.today().to_period('Q').start_time
    q_dates = pd.date_range(start=current_q - pd.DateOffset(months=12),
                            end=current_q + pd.DateOffset(months=12),
                            freq='QS')
    preds = []
    last_hist = df['Date'].max()
    d0 = df['Date'].min()
    for qd in q_dates:
        days_since_start = (qd - d0).days + 1
        pred_price = a_fit * (days_since_start ** slope_fit) if days_since_start > 0 else np.nan
        pred_str = f"{pred_price:,.0f}".replace(",", "'") if np.isfinite(pred_price) else ""
        if qd <= last_hist:
            hist_row = df[df['Date'] <= qd]
            hist_price = hist_row.iloc[-1][price_col] if not hist_row.empty else None
        else:
            hist_price = None
        rel_div = (100 * (hist_price - pred_price) / pred_price) if (hist_price is not None and np.isfinite(pred_price) and pred_price != 0) else None
        preds.append({
            "Quarter": f"{qd.year} Q{qd.quarter}",
            "Days Since Start": int(days_since_start),
            "Predicted Price": pred_str,
            "Historic Price": f"{hist_price:,.0f}".replace(",", "'") if hist_price is not None else "",
            "Relative Divergence (%)": round(rel_div, 2) if rel_div is not None else ""
        })
    df_preds = pd.DataFrame(preds)

    # =========================
    # Build Figure
    # =========================
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.7, 0.3],
                        subplot_titles=("Price Chart", "RSI"))

    # Helper for hover dates
    def dates_h(xvals):
        return dates_for_hover(xvals, use_loglog, base_date)

    # 1) Price - base & segmented by RSI class
    fig.add_trace(
        go.Scatter(
            x=df_chart[x_axis_col],
            y=df_chart[price_col],
            mode='lines',
            line=dict(color=PRICE_COLORS['neutral'], width=price_line_thickness),
            showlegend=False,
            hovertemplate=HOVER_PRICE,
            customdata=dates_h(df_chart[x_axis_col])
        ),
        row=1, col=1
    )

    df_seg = df_chart.sort_values('Date').copy()
    df_seg['Group'] = (df_seg['RSI_class'] != df_seg['RSI_class'].shift()).cumsum()
    for _, g in df_seg.groupby('Group'):
        fig.add_trace(
            go.Scatter(
                x=g[x_axis_col],
                y=g[price_col],
                mode='lines',
                line=dict(color=PRICE_COLORS.get(g['RSI_class'].iloc[0], 'blue')),
                showlegend=False,
                hovertemplate=HOVER_PRICE,
                customdata=dates_h(g[x_axis_col])
            ),
            row=1, col=1
        )

    # 2) MAs
    x_all = df_chart[x_axis_col]
    fig.add_trace(
        go.Scatter(
            x=x_all, y=df_chart['MA_200d'], mode='lines',
            line=dict(color='orange', width=price_line_thickness),
            name=f"Short-Term {fmt_duration_days(short_ma_window)} MA",
            hovertemplate=HOVER_PRICE,
            customdata=dates_h(x_all)
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_all, y=df_chart['MA_200w'], mode='lines',
            line=dict(color='purple', width=price_line_thickness),
            name=f"Long-Term {fmt_duration_days(long_ma_window*7)} MA",
            hovertemplate=HOVER_PRICE,
            customdata=dates_h(x_all)
        ), row=1, col=1
    )

    # 3) Signal bubbles
    # Buy bubbles with performance in hover (humanized duration)
    for df_cat, name, color, symbol in [
        (st_buy, 'Short-term Buy', 'lightgreen', 'circle'),
        (lt_buy, 'Long-term Buy', 'green', 'square'),
    ]:
        if not df_cat.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_cat[x_axis_col], y=df_cat[price_col], mode='markers',
                    marker=dict(size=marker_sizes(df_cat), color=color, opacity=1.0, symbol=symbol),
                    name=name,
                    hovertemplate=HOVER_BUY,
                    customdata=np.column_stack([
                        dates_h(df_cat[x_axis_col]),
                        df_cat['ReturnPct'].fillna(np.nan),
                        df_cat['ARRPct'].fillna(np.nan),
                        df_cat['DaysHeldHuman'].fillna("")
                    ])
                ), row=1, col=1
            )

    # Sell bubbles WITHOUT performance (as requested)
    if not st_sell.empty:
        fig.add_trace(
            go.Scatter(
                x=st_sell[x_axis_col], y=st_sell[price_col], mode='markers',
                marker=dict(size=marker_sizes(st_sell), color='lightcoral', opacity=1.0, symbol='circle'),
                name='Short-term Sell',
                hovertemplate=HOVER_SELL,
                customdata=dates_h(st_sell[x_axis_col])
            ), row=1, col=1
        )

    # 4) Power-law fit line
    if use_loglog:
        x_line = np.linspace(df_chart['Days'].min(), df_chart['Days'].max(), 100)
        y_line = a_fit * (x_line ** slope_fit)
        mask = y_line >= YFIT_MIN_PRICE
        x_line = x_line[mask]; y_line = y_line[mask]
        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line, mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Power-Law Fit',
                hovertemplate=HOVER_PRICE,
                customdata=dates_h(x_line)
            ), row=1, col=1
        )
        fig.update_xaxes(type="log", row=1, col=1)
    else:
        x_line = np.linspace(df['Days'].min(), df['Days'].max(), 100)
        y_line = a_fit * (x_line ** slope_fit)
        mask = y_line >= YFIT_MIN_PRICE
        x_line = x_line[mask]; y_line = y_line[mask]
        x_dates = [df['Date'].min() + pd.Timedelta(days=int(x)-1) for x in x_line]
        fig.add_trace(
            go.Scatter(
                x=x_dates, y=y_line, mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Power-Law Fit',
                hovertemplate=HOVER_PRICE,
                customdata=[d.strftime('%Y-%m-%d') for d in x_dates]
            ), row=1, col=1
        )
        fig.update_xaxes(type="date", row=1, col=1)

    # Y axis mode for price panel
    fig.update_yaxes(type="log" if axis_scale in ['log', 'loglog'] else "linear", row=1, col=1, autorange=True)

    # 5) RSI panel with dynamic zones & segmented colors
    fig.add_trace(
        go.Scatter(
            x=df_chart[x_axis_col], y=df_chart['RSI'], mode='lines',
            line=dict(color='lightgray', width=1), showlegend=False,
            hovertemplate=HOVER_RSI,
            customdata=dates_h(df_chart[x_axis_col])
        ), row=2, col=1
    )
    df_rsi = df_chart.sort_values('Date').copy()
    df_rsi['Group'] = (df_rsi['RSI_class'] != df_rsi['RSI_class'].shift()).cumsum()
    for _, g in df_rsi.groupby('Group'):
        fig.add_trace(
            go.Scatter(
                x=g[x_axis_col], y=g['RSI'], mode='lines',
                line=dict(color=RSI_LINE_COLORS.get(g['RSI_class'].iloc[0], 'black')),
                showlegend=False,
                hovertemplate=HOVER_RSI,
                customdata=dates_h(g[x_axis_col])
            ), row=2, col=1
        )

    x0 = df_chart['Days'].min() if use_loglog else df_chart['Date'].min()
    x1 = df_chart['Days'].max() if use_loglog else df_chart['Date'].max()

    # Dynamic RSI zones
    fig.add_shape(type="rect", xref="x", yref="y2", x0=x0, x1=x1, y0=rsi_ob, y1=100,
                  fillcolor="rgba(255,0,0,0.1)", line_width=0, layer="below", row=2, col=1)
    fig.add_shape(type="rect", xref="x", yref="y2", x0=x0, x1=x1, y0=0, y1=rsi_os,
                  fillcolor="rgba(0,255,0,0.1)", line_width=0, layer="below", row=2, col=1)

    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_xaxes(type="log" if use_loglog else "date", row=2, col=1)

    fig.update_layout(
        height=700,
        title_text=f"{pair} Price & RSI (Coupled X-axis)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Zoom to selected date range + tighten y-range on price
    if start_date and end_date:
        s_date = max(pd.to_datetime(start_date), df['Date'].min())
        e_date = min(pd.to_datetime(end_date), df['Date'].max())
        if use_loglog:
            d0 = df['Date'].min()
            s_day = max((s_date - d0).days + 1, 1)
            e_day = max((e_date - d0).days + 1, s_day + 1)
            fig.update_xaxes(range=[np.log10(s_day), np.log10(e_day)], row=1, col=1)
            fig.update_xaxes(range=[np.log10(s_day), np.log10(e_day)], row=2, col=1)
            subset = df[(df['Days'] >= s_day) & (df['Days'] <= e_day)]
        else:
            fig.update_xaxes(range=[s_date, e_date], row=1, col=1)
            fig.update_xaxes(range=[s_date, e_date], row=2, col=1)
            subset = df[(df['Date'] >= s_date) & (df['Date'] <= e_date)]
        if not subset.empty:
            if axis_scale in ['log', 'loglog']:
                y0, y1 = np.log10(subset[price_col].min()), np.log10(subset[price_col].max())
                pad = (y1 - y0) * 0.05
                fig.update_yaxes(range=[y0 - pad, y1 + pad], row=1, col=1)
            else:
                y0, y1 = subset[price_col].min(), subset[price_col].max()
                pad = (y1 - y0) * 0.05
                fig.update_yaxes(range=[y0 - pad, y1 + pad], row=1, col=1)

    # Debug & strategy text
    debug_text = (f"Pair: {pair} | Data {df['Date'].min().date()} → {df['Date'].max().date()} | "
                  f"Price: {price_field} | ST-MA: {short_ma_window}d | LT-MA: {long_ma_window}w | "
                  f"RSI: oversold < {rsi_os}, overbought > {rsi_ob}")

    # Human-readable power-law header
    pl_readable = fmt_power_law_readable(a_fit, slope_fit)
    q_title = f"Quarterly Price Predictions — Power-Law Fit: {pl_readable}"

    strategy_md = f"""
**Trading Strategy:**  
- **Short-term:** Buy if *{price_field}* < Short-Term MA and RSI < {rsi_os}%; Sell if *{price_field}* > Short-Term MA and RSI > {rsi_ob}%.  
- **Long-term:** Buy if *{price_field}* < Long-Term MA and RSI < {rsi_os}%.  
Short-Term MA = {fmt_duration_days(short_ma_window)}; Long-Term MA = {fmt_duration_days(long_ma_window * 7)}.  
Bubble size ∝ |Price − MA|, max size = 10× line thickness × marker-scale.
"""

    return fig, df_preds.to_dict('records'), q_title, debug_text, strategy_md

# =========================
# Entrypoint
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8050)
