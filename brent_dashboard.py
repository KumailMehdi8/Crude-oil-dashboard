"""
Brent Crude Oil Interactive Dashboard
=====================================
Production-level Dash dashboard for Brent crude oil price analysis (1987–2026).
Includes 10+ interactive filters, multiple chart types, and automated insights.

Requirements:
    pip install dash plotly pandas numpy

Run:
    python brent_dashboard.py
    Open http://127.0.0.1:8050 in your browser
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from datetime import date, timedelta

# ─────────────────────────────────────────────
# 1. DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────

def load_data(path="brent-daily.csv"):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Time features
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["Quarter"]   = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekNum"]   = df["Date"].dt.isocalendar().week.astype(int)

    # Rolling statistics
    df["MA_30"]   = df["Price"].rolling(30).mean()
    df["MA_90"]   = df["Price"].rolling(90).mean()
    df["MA_200"]  = df["Price"].rolling(200).mean()
    df["Volatility_30"] = df["Price"].rolling(30).std()

    # Daily change
    df["DailyChange"]  = df["Price"].diff()
    df["DailyChangePct"] = df["Price"].pct_change() * 100

    # Price regime label
    df["PriceRegime"] = pd.cut(
        df["Price"],
        bins=[0, 30, 60, 90, 120, 999],
        labels=["Very Low (<$30)", "Low ($30–$60)", "Moderate ($60–$90)",
                "High ($90–$120)", "Very High (>$120)"]
    )

    # Decade
    df["Decade"] = (df["Year"] // 10 * 10).astype(str) + "s"

    return df


df = load_data("/mnt/user-data/uploads/brent-daily.csv")

# Useful constants for filter widgets
MIN_DATE   = df["Date"].min().date()
MAX_DATE   = df["Date"].max().date()
MIN_PRICE  = float(df["Price"].min())
MAX_PRICE  = float(df["Price"].max())
ALL_YEARS  = sorted(df["Year"].unique())
DECADES    = sorted(df["Decade"].unique())
QUARTERS   = [1, 2, 3, 4]
MONTHS     = list(range(1, 13))
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
REGIMES    = [str(r) for r in df["PriceRegime"].cat.categories]

# ─────────────────────────────────────────────
# 2. APP INITIALISATION
# ─────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    title="Brent Crude Oil Dashboard",
    suppress_callback_exceptions=True
)

server = app.server

# ─────────────────────────────────────────────
# 3. REUSABLE COMPONENT HELPERS
# ─────────────────────────────────────────────

CARD_STYLE = {
    "background": "#1e2433",
    "border": "1px solid #2d3550",
    "borderRadius": "12px",
    "padding": "16px",
    "marginBottom": "14px"
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0, "left": 0, "bottom": 0,
    "width": "300px",
    "padding": "20px",
    "background": "#141928",
    "overflowY": "auto",
    "borderRight": "1px solid #2d3550",
    "zIndex": 1000
}

CONTENT_STYLE = {
    "marginLeft": "316px",
    "padding": "20px",
    "background": "#0d1117",
    "minHeight": "100vh"
}

ACCENT       = "#f5a623"
ACCENT2      = "#00c8ff"
TEXT_MUTED   = "#8892a4"
CHART_BG     = "#1e2433"
GRID_COLOR   = "#2d3550"

def chart_layout(fig, title="", height=350):
    """Apply consistent dark theme to any figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=14), x=0.01),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color="#c9d1d9"),
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color="#8892a4"),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color="#8892a4"),
    )
    return fig


def filter_label(text):
    return html.Label(text, style={"color": TEXT_MUTED, "fontSize": "11px",
                                    "fontWeight": "600", "letterSpacing": "0.8px",
                                    "textTransform": "uppercase", "marginBottom": "6px"})


def kpi_card(title, value, delta=None, color=ACCENT):
    delta_block = html.Span(
        delta, style={"fontSize": "12px", "color": "#2dce89" if delta and delta.startswith("+") else "#f5365c",
                       "marginLeft": "8px"}
    ) if delta else ""
    return html.Div([
        html.P(title, style={"color": TEXT_MUTED, "fontSize": "11px",
                              "fontWeight": "600", "letterSpacing": "0.8px",
                              "textTransform": "uppercase", "margin": "0 0 4px"}),
        html.Div([
            html.H3(value, style={"color": color, "margin": "0", "display": "inline"}),
            delta_block
        ])
    ], style={**CARD_STYLE, "padding": "14px 18px"})

# ─────────────────────────────────────────────
# 4. SIDEBAR  (10 FILTERS)
# ─────────────────────────────────────────────

sidebar = html.Div([
    # Logo / title
    html.Div([
        html.I(className="fa fa-oil-can", style={"color": ACCENT, "fontSize": "22px", "marginRight": "10px"}),
        html.Span("Brent Crude", style={"color": "white", "fontWeight": "700", "fontSize": "18px"}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
    html.P("Interactive Oil Price Dashboard", style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "24px"}),

    html.Hr(style={"borderColor": "#2d3550", "marginBottom": "20px"}),

    # ── FILTER 1 : Date Range ──────────────────
    filter_label("① Date Range"),
    dcc.DatePickerRange(
        id="filter-date",
        min_date_allowed=MIN_DATE,
        max_date_allowed=MAX_DATE,
        start_date=date(2000, 1, 1),
        end_date=MAX_DATE,
        display_format="DD/MM/YYYY",
        style={"width": "100%", "marginBottom": "16px"},
    ),

    # ── FILTER 2 : Price Range Slider ─────────
    filter_label("② Price Range (USD)"),
    dcc.RangeSlider(
        id="filter-price",
        min=0, max=160, step=5,
        value=[0, 160],
        marks={0:"$0", 40:"$40", 80:"$80", 120:"$120", 160:"$160"},
        tooltip={"placement": "bottom", "always_visible": False},
    ),
    html.Div(style={"marginBottom": "16px"}),

    # ── FILTER 3 : Decade Multi-select ────────
    filter_label("③ Decade"),
    dcc.Dropdown(
        id="filter-decade",
        options=[{"label": d, "value": d} for d in DECADES],
        value=DECADES,
        multi=True,
        style={"marginBottom": "16px"},
        className="dark-dropdown"
    ),

    # ── FILTER 4 : Quarter Multi-select ───────
    filter_label("④ Quarter"),
    dcc.Checklist(
        id="filter-quarter",
        options=[{"label": f" Q{q}", "value": q} for q in QUARTERS],
        value=QUARTERS,
        inline=True,
        inputStyle={"marginRight": "4px"},
        labelStyle={"color": "#c9d1d9", "marginRight": "12px", "fontSize": "13px"},
        style={"marginBottom": "16px"}
    ),

    # ── FILTER 5 : Month Multi-select ─────────
    filter_label("⑤ Month"),
    dcc.Dropdown(
        id="filter-month",
        options=[{"label": MONTH_NAMES[m], "value": m} for m in MONTHS],
        value=MONTHS,
        multi=True,
        style={"marginBottom": "16px"},
        className="dark-dropdown"
    ),

    # ── FILTER 6 : Price Regime ────────────────
    filter_label("⑥ Price Regime"),
    dcc.Dropdown(
        id="filter-regime",
        options=[{"label": r, "value": r} for r in REGIMES],
        value=REGIMES,
        multi=True,
        style={"marginBottom": "16px"},
        className="dark-dropdown"
    ),

    # ── FILTER 7 : Moving Average Overlay ─────
    filter_label("⑦ MA Overlay"),
    dcc.Checklist(
        id="filter-ma",
        options=[
            {"label": " 30-day MA",  "value": "MA_30"},
            {"label": " 90-day MA",  "value": "MA_90"},
            {"label": " 200-day MA", "value": "MA_200"},
        ],
        value=["MA_90"],
        inputStyle={"marginRight": "4px"},
        labelStyle={"color": "#c9d1d9", "marginRight": "10px", "fontSize": "13px",
                    "display": "block", "marginBottom": "4px"},
        style={"marginBottom": "16px"}
    ),

    # ── FILTER 8 : Top-N years ────────────────
    filter_label("⑧ Show Top-N Years (by avg price)"),
    dcc.Slider(
        id="filter-topn",
        min=5, max=len(ALL_YEARS), step=5,
        value=len(ALL_YEARS),
        marks={5:"5", 15:"15", 25:"25", len(ALL_YEARS): "All"},
        tooltip={"placement": "bottom", "always_visible": False},
    ),
    html.Div(style={"marginBottom": "16px"}),

    # ── FILTER 9 : Volatility Threshold ───────
    filter_label("⑨ Daily Change Threshold (%)"),
    dcc.Slider(
        id="filter-vol",
        min=0, max=20, step=1,
        value=0,
        marks={0:"0%", 5:"5%", 10:"10%", 15:"15%", 20:"20%"},
        tooltip={"placement": "bottom", "always_visible": False},
    ),
    html.Div(style={"marginBottom": "16px"}),

    # ── FILTER 10 : Text / Year Search ────────
    filter_label("⑩ Jump to Year"),
    dcc.Input(
        id="filter-year-text",
        type="number",
        placeholder="e.g. 2008",
        min=int(df["Year"].min()), max=int(df["Year"].max()),
        debounce=True,
        style={"width": "100%", "marginBottom": "16px",
               "background": "#252d3d", "color": "white",
               "border": "1px solid #2d3550", "borderRadius": "6px",
               "padding": "8px 10px"},
    ),

    html.Hr(style={"borderColor": "#2d3550"}),
    html.Button(
        [html.I(className="fa fa-rotate-left", style={"marginRight": "8px"}), "Reset All Filters"],
        id="btn-reset",
        n_clicks=0,
        style={"width": "100%", "background": "#252d3d", "color": ACCENT,
               "border": f"1px solid {ACCENT}", "borderRadius": "8px",
               "padding": "10px", "cursor": "pointer", "fontWeight": "600",
               "marginTop": "8px"}
    ),
], style=SIDEBAR_STYLE)

# ─────────────────────────────────────────────
# 5. MAIN CONTENT LAYOUT
# ─────────────────────────────────────────────

tab_style     = {"color": TEXT_MUTED, "backgroundColor": "transparent",
                  "borderTop": "none", "borderLeft": "none", "borderRight": "none",
                  "borderBottom": "2px solid transparent", "padding": "10px 20px"}
tab_sel_style = {**tab_style, "color": ACCENT, "borderBottom": f"2px solid {ACCENT}",
                  "fontWeight": "700"}

content = html.Div([
    # ── Header ───────────────────────────────
    html.Div([
        html.H2("🛢️ Brent Crude Oil Price Analytics", style={"color": "white", "margin": "0 0 4px"}),
        html.P("Daily spot price data · 1987–2026 · Source: EIA / World Bank",
               style={"color": TEXT_MUTED, "fontSize": "13px"}),
    ], style={"marginBottom": "20px"}),

    # ── KPI Row ───────────────────────────────
    html.Div(id="kpi-row", style={"display": "grid",
                                   "gridTemplateColumns": "repeat(5, 1fr)",
                                   "gap": "12px", "marginBottom": "20px"}),

    # ── Tabs ──────────────────────────────────
    dcc.Tabs(id="main-tabs", value="overview", children=[
        dcc.Tab(label="📈 Overview",    value="overview",    style=tab_style, selected_style=tab_sel_style),
        dcc.Tab(label="📊 Trends",      value="trends",      style=tab_style, selected_style=tab_sel_style),
        dcc.Tab(label="🔁 Comparisons", value="comparisons", style=tab_style, selected_style=tab_sel_style),
        dcc.Tab(label="🔍 Deep Dive",   value="deep",        style=tab_style, selected_style=tab_sel_style),
        dcc.Tab(label="💡 Insights",    value="insights",    style=tab_style, selected_style=tab_sel_style),
    ], style={"borderBottom": "1px solid #2d3550", "marginBottom": "20px"}),

    html.Div(id="tab-content"),

], style=CONTENT_STYLE)

# ─────────────────────────────────────────────
# 6. APP LAYOUT
# ─────────────────────────────────────────────

app.layout = html.Div([sidebar, content], style={"background": "#0d1117", "minHeight": "100vh"})

# ─────────────────────────────────────────────
# 7. HELPER: APPLY ALL FILTERS
# ─────────────────────────────────────────────

def apply_filters(start_date, end_date, price_range, decades, quarters, months,
                  regimes, topn, vol_threshold, year_jump):
    dff = df.copy()

    # Date range
    if start_date: dff = dff[dff["Date"] >= pd.Timestamp(start_date)]
    if end_date:   dff = dff[dff["Date"] <= pd.Timestamp(end_date)]

    # Price range
    dff = dff[(dff["Price"] >= price_range[0]) & (dff["Price"] <= price_range[1])]

    # Decade
    if decades: dff = dff[dff["Decade"].isin(decades)]

    # Quarter
    if quarters: dff = dff[dff["Quarter"].isin(quarters)]

    # Month
    if months: dff = dff[dff["Month"].isin(months)]

    # Price regime
    if regimes: dff = dff[dff["PriceRegime"].astype(str).isin(regimes)]

    # Top-N years by average price
    if topn and topn < len(ALL_YEARS):
        top_years = (dff.groupby("Year")["Price"].mean()
                       .nlargest(topn).index.tolist())
        dff = dff[dff["Year"].isin(top_years)]

    # Volatility threshold: keep only days with |DailyChangePct| >= threshold
    if vol_threshold and vol_threshold > 0:
        dff = dff[dff["DailyChangePct"].abs() >= vol_threshold]

    # Year jump: narrow to ±2 years
    if year_jump and str(year_jump).isdigit():
        y = int(year_jump)
        dff = dff[(dff["Year"] >= y-2) & (dff["Year"] <= y+2)]

    return dff

# ─────────────────────────────────────────────
# 8. RESET CALLBACK
# ─────────────────────────────────────────────

@app.callback(
    Output("filter-date",     "start_date"),
    Output("filter-date",     "end_date"),
    Output("filter-price",    "value"),
    Output("filter-decade",   "value"),
    Output("filter-quarter",  "value"),
    Output("filter-month",    "value"),
    Output("filter-regime",   "value"),
    Output("filter-topn",     "value"),
    Output("filter-vol",      "value"),
    Output("filter-year-text","value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(_):
    return (date(2000, 1, 1), MAX_DATE, [0, 160], DECADES, QUARTERS,
            MONTHS, REGIMES, len(ALL_YEARS), 0, None)

# ─────────────────────────────────────────────
# 9. KPI CALLBACK
# ─────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Input("filter-date",     "start_date"),
    Input("filter-date",     "end_date"),
    Input("filter-price",    "value"),
    Input("filter-decade",   "value"),
    Input("filter-quarter",  "value"),
    Input("filter-month",    "value"),
    Input("filter-regime",   "value"),
    Input("filter-topn",     "value"),
    Input("filter-vol",      "value"),
    Input("filter-year-text","value"),
)
def update_kpis(sd, ed, pr, dec, qtr, mon, reg, tn, vl, yr):
    dff = apply_filters(sd, ed, pr, dec, qtr, mon, reg, tn, vl, yr)
    if dff.empty:
        return [html.P("No data for selected filters.", style={"color": "red"})]

    cur_price  = dff.iloc[-1]["Price"]
    prev_price = dff.iloc[-2]["Price"] if len(dff) > 1 else cur_price
    chg_pct    = (cur_price - prev_price) / prev_price * 100
    avg_price  = dff["Price"].mean()
    max_price  = dff["Price"].max()
    min_price  = dff["Price"].min()
    volatility = dff["DailyChangePct"].std()

    sign = "+" if chg_pct >= 0 else ""

    return [
        kpi_card("Latest Price",   f"${cur_price:.2f}",  f"{sign}{chg_pct:.2f}%", ACCENT),
        kpi_card("Average Price",  f"${avg_price:.2f}",  color=ACCENT2),
        kpi_card("Max Price",      f"${max_price:.2f}",  color="#f5365c"),
        kpi_card("Min Price",      f"${min_price:.2f}",  color="#2dce89"),
        kpi_card("Volatility (σ)", f"{volatility:.2f}%", color="#b39ddb"),
    ]

# ─────────────────────────────────────────────
# 10. TAB CONTENT CALLBACK
# ─────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs",       "value"),
    Input("filter-date",     "start_date"),
    Input("filter-date",     "end_date"),
    Input("filter-price",    "value"),
    Input("filter-decade",   "value"),
    Input("filter-quarter",  "value"),
    Input("filter-month",    "value"),
    Input("filter-regime",   "value"),
    Input("filter-topn",     "value"),
    Input("filter-vol",      "value"),
    Input("filter-year-text","value"),
    Input("filter-ma",       "value"),
)
def update_tabs(tab, sd, ed, pr, dec, qtr, mon, reg, tn, vl, yr, ma_lines):
    dff = apply_filters(sd, ed, pr, dec, qtr, mon, reg, tn, vl, yr)

    if dff.empty:
        return html.Div("⚠️  No data matches the current filters. Try widening the selection.",
                        style={"color": "#f5365c", "padding": "40px", "textAlign": "center",
                               "fontSize": "16px"})

    # ── OVERVIEW TAB ─────────────────────────
    if tab == "overview":
        # Price line + MA overlays
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=dff["Date"], y=dff["Price"], name="Brent Price",
            line=dict(color=ACCENT, width=1.5), mode="lines"
        ))
        ma_colors = {"MA_30": "#00c8ff", "MA_90": "#2dce89", "MA_200": "#b39ddb"}
        ma_labels = {"MA_30": "30-Day MA", "MA_90": "90-Day MA", "MA_200": "200-Day MA"}
        for col in (ma_lines or []):
            if col in dff.columns:
                fig_line.add_trace(go.Scatter(
                    x=dff["Date"], y=dff[col], name=ma_labels[col],
                    line=dict(color=ma_colors[col], width=1.5, dash="dot"), mode="lines"
                ))
        chart_layout(fig_line, "Brent Crude Oil Price (USD/bbl)", height=380)

        # Volume-of-change histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=dff["DailyChangePct"].dropna(), nbinsx=80,
            marker_color=ACCENT, opacity=0.75, name="Daily % Change"
        ))
        chart_layout(fig_hist, "Distribution of Daily % Changes", height=280)

        return html.Div([
            html.Div(dcc.Graph(figure=fig_line), style=CARD_STYLE),
            html.Div(dcc.Graph(figure=fig_hist), style=CARD_STYLE),
        ])

    # ── TRENDS TAB ───────────────────────────
    elif tab == "trends":
        # Yearly average bar
        yearly = dff.groupby("Year")["Price"].agg(["mean","max","min"]).reset_index()
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=yearly["Year"], y=yearly["mean"],
                                  name="Avg Price", marker_color=ACCENT, opacity=0.85))
        fig_bar.add_trace(go.Scatter(x=yearly["Year"], y=yearly["max"],
                                      name="Max Price", line=dict(color="#f5365c"), mode="lines+markers"))
        fig_bar.add_trace(go.Scatter(x=yearly["Year"], y=yearly["min"],
                                      name="Min Price", line=dict(color="#2dce89"), mode="lines+markers"))
        chart_layout(fig_bar, "Annual Average / Max / Min Price", height=320)

        # Monthly seasonality
        monthly = dff.groupby("Month")["Price"].mean().reset_index()
        monthly["MonthName"] = monthly["Month"].map(MONTH_NAMES)
        fig_season = go.Figure(go.Bar(
            x=monthly["MonthName"], y=monthly["Price"],
            marker_color=[ACCENT if p >= monthly["Price"].mean() else ACCENT2 for p in monthly["Price"]]
        ))
        chart_layout(fig_season, "Average Price by Month (Seasonality)", height=280)

        # Quarterly box plot
        fig_box = go.Figure()
        for q in sorted(dff["Quarter"].unique()):
            subset = dff[dff["Quarter"] == q]["Price"]
            fig_box.add_trace(go.Box(y=subset, name=f"Q{q}",
                                      marker_color=[ACCENT, ACCENT2, "#f5365c", "#2dce89"][q-1]))
        chart_layout(fig_box, "Price Distribution by Quarter", height=280)

        return html.Div([
            html.Div(dcc.Graph(figure=fig_bar),    style=CARD_STYLE),
            html.Div([
                html.Div(dcc.Graph(figure=fig_season), style={**CARD_STYLE, "display":"inline-block","width":"49%"}),
                html.Div(dcc.Graph(figure=fig_box),    style={**CARD_STYLE, "display":"inline-block","width":"49%","marginLeft":"2%"}),
            ]),
        ])

    # ── COMPARISONS TAB ──────────────────────
    elif tab == "comparisons":
        # Price regime pie
        regime_counts = dff["PriceRegime"].value_counts().reset_index()
        regime_counts.columns = ["Regime", "Count"]
        fig_pie = go.Figure(go.Pie(
            labels=regime_counts["Regime"], values=regime_counts["Count"],
            hole=0.4, marker=dict(colors=[ACCENT, ACCENT2, "#2dce89", "#f5365c", "#b39ddb"]),
            textinfo="label+percent"
        ))
        chart_layout(fig_pie, "Days by Price Regime", height=320)

        # Decade comparison
        decade_avg = dff.groupby("Decade")["Price"].mean().reset_index()
        fig_dec = go.Figure(go.Bar(
            x=decade_avg["Decade"], y=decade_avg["Price"],
            marker_color=ACCENT, text=decade_avg["Price"].round(1), textposition="outside",
            textfont=dict(color="white")
        ))
        chart_layout(fig_dec, "Average Price by Decade", height=280)

        # Year-over-year % change
        yearly2 = dff.groupby("Year")["Price"].mean().reset_index()
        yearly2["YoY"] = yearly2["Price"].pct_change() * 100
        fig_yoy = go.Figure(go.Bar(
            x=yearly2["Year"], y=yearly2["YoY"],
            marker_color=["#2dce89" if v >= 0 else "#f5365c" for v in yearly2["YoY"]]
        ))
        chart_layout(fig_yoy, "Year-over-Year % Change in Average Price", height=280)

        return html.Div([
            html.Div([
                html.Div(dcc.Graph(figure=fig_pie), style={**CARD_STYLE,"display":"inline-block","width":"35%"}),
                html.Div(dcc.Graph(figure=fig_dec), style={**CARD_STYLE,"display":"inline-block","width":"62%","marginLeft":"3%"}),
            ]),
            html.Div(dcc.Graph(figure=fig_yoy), style=CARD_STYLE),
        ])

    # ── DEEP DIVE TAB ────────────────────────
    elif tab == "deep":
        # Scatter: price vs daily change
        sample = dff.sample(min(3000, len(dff)), random_state=42)
        fig_scatter = go.Figure(go.Scatter(
            x=sample["Price"], y=sample["DailyChangePct"],
            mode="markers",
            marker=dict(color=sample["Year"], colorscale="Plasma",
                        size=4, opacity=0.6, colorbar=dict(title="Year")),
            text=sample["Date"].dt.strftime("%Y-%m-%d"),
            hovertemplate="Date: %{text}<br>Price: $%{x:.2f}<br>Change: %{y:.2f}%<extra></extra>"
        ))
        chart_layout(fig_scatter, "Price vs Daily % Change (coloured by year)", height=320)

        # Rolling 30-day volatility
        dff2 = dff.copy()
        dff2["Volatility_30"] = dff2["Price"].rolling(30).std()
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=dff2["Date"], y=dff2["Volatility_30"], fill="tozeroy",
            line=dict(color="#b39ddb"), name="30-Day Volatility"
        ))
        chart_layout(fig_vol, "30-Day Rolling Price Volatility (Std Dev)", height=280)

        # Heatmap: avg price by Year × Month
        pivot = dff.pivot_table(index="Year", columns="Month", values="Price", aggfunc="mean")
        pivot.columns = [MONTH_NAMES[c] for c in pivot.columns]
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale="Plasma", colorbar=dict(title="USD/bbl")
        ))
        chart_layout(fig_heat, "Average Price Heatmap (Year × Month)", height=360)

        return html.Div([
            html.Div(dcc.Graph(figure=fig_scatter), style=CARD_STYLE),
            html.Div(dcc.Graph(figure=fig_vol),     style=CARD_STYLE),
            html.Div(dcc.Graph(figure=fig_heat),    style=CARD_STYLE),
        ])

    # ── INSIGHTS TAB ─────────────────────────
    elif tab == "insights":
        # Compute insights from filtered data
        avg   = dff["Price"].mean()
        med   = dff["Price"].median()
        std   = dff["Price"].std()
        skew  = dff["Price"].skew()
        peak_row  = dff.loc[dff["Price"].idxmax()]
        trough_row = dff.loc[dff["Price"].idxmin()]
        best_year  = dff.groupby("Year")["Price"].mean().idxmax()
        worst_year = dff.groupby("Year")["Price"].mean().idxmin()
        most_vol   = dff.groupby("Year")["DailyChangePct"].std().idxmax()

        # Count extreme days
        extreme_up   = (dff["DailyChangePct"] >  5).sum()
        extreme_down = (dff["DailyChangePct"] < -5).sum()

        def insight_card(icon, title, body):
            return html.Div([
                html.Div([html.Span(icon, style={"fontSize":"22px","marginRight":"12px"}),
                          html.Strong(title, style={"color": "white"})],
                         style={"display":"flex","alignItems":"center","marginBottom":"6px"}),
                html.P(body, style={"color":"#c9d1d9","margin":"0","fontSize":"13px","lineHeight":"1.5"})
            ], style={**CARD_STYLE,"borderLeft":f"3px solid {ACCENT}"})

        return html.Div([
            html.H4("Automated Insights", style={"color": "white", "marginBottom": "16px"}),
            html.P(f"Based on {len(dff):,} trading days in the filtered dataset.",
                   style={"color": TEXT_MUTED, "marginBottom": "20px"}),
            insight_card("📊", "Price Statistics",
                f"Average: ${avg:.2f} | Median: ${med:.2f} | Std Dev: ${std:.2f}. "
                f"The distribution is {'right-skewed' if skew>0.3 else 'left-skewed' if skew<-0.3 else 'roughly symmetric'} (skewness: {skew:.2f})."),
            insight_card("🏔️", "All-Time Peak",
                f"Highest price: ${peak_row['Price']:.2f}/bbl on {peak_row['Date'].strftime('%d %b %Y')}. "
                f"This was during {peak_row['Year']}, a period of global supply tension."),
            insight_card("📉", "All-Time Trough",
                f"Lowest price: ${trough_row['Price']:.2f}/bbl on {trough_row['Date'].strftime('%d %b %Y')}. "
                f"Markets were under significant supply/demand pressure at this point."),
            insight_card("📅", "Best & Worst Annual Averages",
                f"{best_year} had the highest average annual price; "
                f"{worst_year} had the lowest. Year-over-year swings highlight the impact of geopolitical and macro events."),
            insight_card("⚡", "Most Volatile Year",
                f"{most_vol} was the most volatile year in the filtered range, with the highest standard deviation in daily % changes — "
                f"likely driven by a major geopolitical shock or demand collapse."),
            insight_card("🚨", "Extreme Price Moves",
                f"There were {extreme_up} days with a single-day gain >5% and {extreme_down} days "
                f"with a single-day loss >5%. These tail events are important risk indicators."),
            insight_card("🔄", "Price Regime Breakdown",
                ", ".join([f"{r}: {(dff['PriceRegime'].astype(str)==r).sum():,} days"
                           for r in REGIMES if (dff['PriceRegime'].astype(str)==r).any()])),
        ])

    return html.Div("Select a tab.")

# ─────────────────────────────────────────────
# 11. RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🛢️  Brent Crude Oil Dashboard")
    print("  Open:  http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=False, port=8050)
