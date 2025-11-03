# Webstock_v6.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import calendar
import re
import altair as alt

# -----------------------------
# Helpers
# -----------------------------
def normalize_numeric(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return np.nan
    s = str(x).strip().replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?=\s|$)", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan

def load_colmex_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=";", engine="python")
    for col in ["Gross P/L", "Execution fee", "Net P/L", "Price", "Quantity"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_numeric)

    if "Date/Time" not in df.columns:
        raise ValueError("Missing 'Date/Time' column in the CSV.")

    df = df.copy()
    df["Date/Time"] = df["Date/Time"].astype(str).str.replace("\xa0", " ").str.strip()
    df["Date/Time"] = df["Date/Time"].str.replace(r"\s+", " ", regex=True)

    df["datetime"] = pd.to_datetime(df["Date/Time"], dayfirst=True, errors="coerce")
    if df["datetime"].isna().all():
        raise ValueError("Could not parse 'Date/Time'. Ensure day-first format like 29.10.2025 15:55:13")
    df["date"] = df["datetime"].dt.floor("D")

    if "Net P/L" not in df.columns or df["Net P/L"].isna().all():
        net = None
        if "Gross P/L" in df.columns and "Execution fee" in df.columns:
            net = df["Gross P/L"].fillna(0) - df["Execution fee"].fillna(0)
        df["Net P/L"] = net

    fees = df["Execution fee"] if "Execution fee" in df.columns else 0.0
    df["__fees__"] = pd.to_numeric(fees, errors="coerce").fillna(0.0).astype(float)
    df["Net P/L"] = pd.to_numeric(df["Net P/L"], errors="coerce").fillna(0.0).astype(float)
    return df

def build_daily_pnl(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("date", as_index=False).agg(
        pnl=("Net P/L", "sum"),
        fees=("__fees__", "sum"),
        trades=("Symbol", "count") if "Symbol" in df.columns else ("date", "count")
    )
    g["pnl"] = pd.to_numeric(g["pnl"], errors="coerce").fillna(0.0).astype(float)
    g["fees"] = pd.to_numeric(g["fees"], errors="coerce").fillna(0.0).astype(float)
    return g.sort_values("date")

def style_currency(x):
    try:
        v = float(x)
    except Exception:
        return ""
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"

def month_bounds(month_start: date):
    last_day = calendar.monthrange(month_start.year, month_start.month)[1]
    start = date(month_start.year, month_start.month, 1)
    end = date(month_start.year, month_start.month, last_day)
    return start, end

def month_weeks_sunday_first(month_start: date):
    cal = calendar.Calendar(firstweekday=6)  # Sunday
    return cal.monthdatescalendar(month_start.year, month_start.month)

def sunday_week_bounds(d: date):
    sunday_offset = (d.weekday() + 1) % 7  # Sun->0, Mon->1,...
    start = d - timedelta(days=sunday_offset)
    end = start + timedelta(days=6)
    return start, end

# Winners helpers
def winners_by_day_from_raw(raw: pd.DataFrame) -> dict:
    if "Symbol" not in raw.columns:
        return {}
    tmp = (raw.groupby(["date","Symbol"], as_index=False)["Net P/L"].sum())
    tmp = tmp[tmp["Net P/L"] > 0]
    winners = (tmp.groupby("date")["Symbol"]
                 .apply(lambda s: ", ".join(sorted(set(s))))
                 .to_dict())
    return winners

def winners_by_week_from_raw(raw: pd.DataFrame) -> dict:
    if "Symbol" not in raw.columns:
        return {}
    # Week ends Saturday; compute week_end for each trade
    wd = raw["date"].dt.weekday  # Mon 0 ... Sun 6
    # Days until Saturday (5)
    days_to_sat = (5 - wd) % 7
    week_end = raw["date"] + pd.to_timedelta(days_to_sat, unit="D")
    tmp = raw.copy()
    tmp["week_end"] = week_end
    g = (tmp.groupby(["week_end","Symbol"], as_index=False)["Net P/L"].sum())
    g = g[g["Net P/L"] > 0]
    winners = (g.groupby("week_end")["Symbol"]
                 .apply(lambda s: ", ".join(sorted(set(s))))
                 .to_dict())
    return winners

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Colmex P&L Calendar", page_icon="📆", layout="wide")
st.title("📆 Colmex Pro — P&L Calendar (Clickable Days)")

uploaded = st.file_uploader("Upload your Colmex 'Filled orders' CSV", type=["csv"])

@st.cache_data(show_spinner=False)
def _load(file) -> pd.DataFrame:
    return load_colmex_csv(file)

if not uploaded:
    st.info("Upload your **Filled orders** CSV to begin.")
    st.stop()

try:
    raw = _load(uploaded)
except Exception as e:
    st.error(f"Failed to parse the CSV. Details: {e}")
    st.stop()

if raw.empty:
    st.warning("No rows found after parsing. Check your file.")
    st.stop()

daily = build_daily_pnl(raw)
min_day = daily["date"].min().date()
max_day = daily["date"].max().date()

# Precompute winners maps
day_winners = winners_by_day_from_raw(raw)
week_winners = winners_by_week_from_raw(raw)

# -----------------------------
# Session state
# -----------------------------
if "current_month" not in st.session_state:
    st.session_state.current_month = date(max_day.year, max_day.month, 1)
if "selected_day" not in st.session_state:
    st.session_state.selected_day = max_day

current_month = st.session_state.current_month
selected_day = st.session_state.selected_day

# Month navigation
nav_l, nav_c, nav_r = st.columns([1,2,1])
with nav_l:
    if st.button("◀ Prev", key="nav_prev"):
        y, m = current_month.year, current_month.month
        if m == 1:
            y, m = y - 1, 12
        else:
            m -= 1
        st.session_state.current_month = date(y, m, 1)

with nav_c:
    st.markdown(f"### {current_month.strftime('%B %Y')}")

with nav_r:
    if st.button("Next ▶", key="nav_next"):
        y, m = current_month.year, current_month.month
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
        st.session_state.current_month = date(y, m, 1)

# Current month slice
m_start, m_end = month_bounds(st.session_state.current_month)
month_mask = (daily["date"].dt.date >= m_start) & (daily["date"].dt.date <= m_end)
month_df = daily.loc[month_mask].copy()

# Maps for calendar bubbles
pnl_map = {d.date(): float(p) for d, p in zip(month_df["date"], month_df["pnl"])}
trd_map = {d.date(): int(t) for d, t in zip(month_df["date"], month_df["trades"])}
fee_map = {d.date(): float(f) for d, f in zip(month_df["date"], month_df["fees"])}

# Weekdays header
WEEKDAYS = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
header_cols = st.columns(7)
for i, name in enumerate(WEEKDAYS):
    with header_cols[i]:
        st.markdown(f"<div style='text-align:center;font-weight:600;'>{name}</div>", unsafe_allow_html=True)

# Calendar grid (buttons per day)
day_rows = month_weeks_sunday_first(st.session_state.current_month)
st.write("")
month_key = st.session_state.current_month.strftime("%Y-%m")
for week in day_rows:
    cols = st.columns(7)
    for idx, d in enumerate(week):
        with cols[idx]:
            if d.month != st.session_state.current_month.month:
                st.markdown("<div style='height:2.2rem;'></div>", unsafe_allow_html=True)
                continue
            pnl = pnl_map.get(d, 0.0)
            trades = trd_map.get(d, 0)
            fees = fee_map.get(d, 0.0)
            icon = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
            label = f"{d.day:02d} {icon}\n{style_currency(pnl)}"
            help_txt = f"Date: {d.isoformat()} | P&L: {style_currency(pnl)} | Fees: {style_currency(fees)} | Trades: {trades}"
            if st.button(label, key=f"day_{month_key}_{d.isoformat()}", help=help_txt):
                st.session_state.selected_day = d
                selected_day = d

# -----------------------------
# Graph: style toggle + rich tooltips
# -----------------------------
st.subheader("Performance Graph")

def alt_theme():
    return {
        "config": {
            "view": {"continuousWidth": 780, "continuousHeight": 280},
            "axis": {"labelFontSize": 12, "titleFontSize": 12, "grid": False},
            "legend": {"labelFontSize": 12, "titleFontSize": 12}
        }
    }
alt.themes.register("minimal_trade_theme", alt_theme)
alt.themes.enable("minimal_trade_theme")

view = st.selectbox("View", ["Daily", "Weekly", "Monthly", "Yearly"], index=0)
style = st.radio("Graph style", ["Line + dots", "Bars"], horizontal=True)

def render_chart(df, x_field, x_title, tooltips):
    if df.empty:
        st.info("No data for this selection.")
        return
    base = alt.Chart(df).encode(
        x=alt.X(f"{x_field}", title=x_title),
        y=alt.Y("pnl:Q", title="P&L ($)"),
        tooltip=tooltips
    )
    if style == "Bars":
        chart = base.mark_bar()
    else:
        chart = base.mark_line(size=1) + base.mark_point(size=60, filled=True)
    st.altair_chart(chart.interactive(), use_container_width=True)

# DAILY — x = trade time, tooltip = trade details
if view == "Daily":
    dtr = raw.loc[raw["date"].dt.date == selected_day].copy()
    if dtr.empty:
        st.info("No trades on the selected day.")
    else:
        dtr = dtr.sort_values("datetime").reset_index(drop=True)
        dtr["cum_pnl"] = dtr["Net P/L"].cumsum()
        cols = ["datetime","cum_pnl","Net P/L","Symbol","Side","Quantity","Price","Description"]
        tooltips = []
        for c in cols:
            if c in dtr.columns:
                if c in ["cum_pnl","Net P/L","Price"]:
                    tooltips.append(alt.Tooltip(c, title=c.replace("_"," ").title(), format="$.2f"))
                else:
                    tooltips.append(alt.Tooltip(c, title=c))
        dplot = dtr.rename(columns={"cum_pnl":"pnl"})
        render_chart(dplot, "datetime:T", "Time", tooltips)

# WEEKLY — Sunday→Saturday, fill missing days; tooltip shows winners
elif view == "Weekly":
    w_start, w_end = sunday_week_bounds(selected_day)
    rng = pd.date_range(w_start, w_end, freq="D")
    wk = daily.loc[(daily["date"].dt.date >= w_start) & (daily["date"].dt.date <= w_end)][["date","pnl","fees","trades"]].copy()
    wk = wk.set_index("date").reindex(rng, fill_value=0.0).rename_axis("date").reset_index()
    # winners per day
    wk["winners"] = wk["date"].map(day_winners).fillna("—")
    tooltips = [
        alt.Tooltip("winners:N", title="Winners"),
        alt.Tooltip("pnl:Q", title="P&L", format="$.2f"),
    ]
    if "fees" in wk.columns: tooltips.append(alt.Tooltip("fees:Q", title="Fees", format="$.2f"))
    if "trades" in wk.columns: tooltips.append(alt.Tooltip("trades:Q", title="Trades"))
    render_chart(wk.rename(columns={"date":"x"}), "x:T", f"Week {w_start.isoformat()} → {w_end.isoformat()}", tooltips)

# MONTHLY — all days in current month, tooltip shows winners
elif view == "Monthly":
    m0, m1 = month_bounds(current_month)
    rng = pd.date_range(m0, m1, freq="D")
    md = month_df[["date","pnl","fees","trades"]].copy()
    md = md.set_index("date").reindex(rng, fill_value=0.0).rename_axis("date").reset_index()
    md["winners"] = md["date"].map(day_winners).fillna("—")
    tooltips = [
        alt.Tooltip("winners:N", title="Winners"),
        alt.Tooltip("pnl:Q", title="P&L", format="$.2f"),
    ]
    if "fees" in md.columns: tooltips.append(alt.Tooltip("fees:Q", title="Fees", format="$.2f"))
    if "trades" in md.columns: tooltips.append(alt.Tooltip("trades:Q", title="Trades"))
    render_chart(md.rename(columns={"date":"x"}), "x:T", current_month.strftime("Days of %B %Y"), tooltips)

# YEARLY — aggregate by week (Sun→Sat), tooltip shows weekly winners
elif view == "Yearly":
    year = selected_day.year
    y0 = date(year, 1, 1); y1 = date(year, 12, 31)
    yr = daily.loc[(daily["date"].dt.date >= y0) & (daily["date"].dt.date <= y1)][["date","pnl","fees"]].copy()
    # Weekly resample ending on Saturday
    yr = (yr.set_index("date")
            .resample("W-SAT").sum()
            .rename_axis("week_end")
            .reset_index())
    # ensure full continuous weekly index
    full_weeks = pd.date_range(pd.Timestamp(y0), pd.Timestamp(y1), freq="W-SAT")
    yr = yr.set_index("week_end").reindex(full_weeks, fill_value=0.0).rename_axis("week_end").reset_index()
    yr["week_start"] = yr["week_end"] - pd.Timedelta(days=6)
    # attach winners by week
    yr["winners"] = yr["week_end"].map(week_winners).fillna("—")
    yr = yr.rename(columns={"week_end":"x"})
    tooltips = [
        alt.Tooltip("winners:N", title="Winners"),
        alt.Tooltip("pnl:Q", title="P&L", format="$.2f"),
    ]
    if "fees" in yr.columns: tooltips.append(alt.Tooltip("fees:Q", title="Fees", format="$.2f"))
    render_chart(yr, "x:T", f"Weeks of {year}", tooltips)
