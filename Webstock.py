# Webstock_v8.py
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
    if "Gross P/L" in df.columns:
        df["Gross P/L"] = pd.to_numeric(df["Gross P/L"], errors="coerce").fillna(0.0).astype(float)
    else:
        df["Gross P/L"] = df["Net P/L"] + df["__fees__"]
    return df

def build_daily(df: pd.DataFrame, pnl_col: str) -> pd.DataFrame:
    g = df.groupby("date", as_index=False).agg(
        pnl=(pnl_col, "sum"),
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

# Winners/Losers helpers (by Net P/L sign)
def _names_for_sign(series: pd.Series) -> str:
    s = sorted(set(series.dropna().tolist()))
    return ", ".join(s) if s else "—"

def winners_losers_by_day(raw: pd.DataFrame):
    if "Symbol" not in raw.columns:
        return {}, {}
    tmp = raw.groupby(["date","Symbol"], as_index=False)["Net P/L"].sum()
    winners = tmp[tmp["Net P/L"] > 0].groupby("date")["Symbol"].apply(_names_for_sign).to_dict()
    losers  = tmp[tmp["Net P/L"] < 0].groupby("date")["Symbol"].apply(_names_for_sign).to_dict()
    return winners, losers

def winners_losers_by_week(raw: pd.DataFrame):
    if "Symbol" not in raw.columns:
        return {}, {}
    wd = raw["date"].dt.weekday
    days_to_sat = (5 - wd) % 7
    week_end = raw["date"] + pd.to_timedelta(days_to_sat, unit="D")
    tmp = raw.copy()
    tmp["week_end"] = week_end
    g = tmp.groupby(["week_end","Symbol"], as_index=False)["Net P/L"].sum()
    winners = g[g["Net P/L"] > 0].groupby("week_end")["Symbol"].apply(_names_for_sign).to_dict()
    losers  = g[g["Net P/L"] < 0].groupby("week_end")["Symbol"].apply(_names_for_sign).to_dict()
    return winners, losers

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

# Precompute winners/losers (by Net P/L)
day_winners, day_losers   = winners_losers_by_day(raw)
week_winners, week_losers = winners_losers_by_week(raw)

# Data for calendar (Net P&L)
daily_net = build_daily(raw, "Net P/L")
min_day = daily_net["date"].min().date()
max_day = daily_net["date"].max().date()

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
month_mask = (daily_net["date"].dt.date >= m_start) & (daily_net["date"].dt.date <= m_end)
month_df = daily_net.loc[month_mask].copy()

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

# ---------- Day details panel (RESTORED) ----------
left, right = st.columns([3,2], vertical_alignment="top")
with left:
    st.subheader(f"Trades on {selected_day}")
    day_trades = raw.loc[raw["date"].dt.date == selected_day].copy()
    if day_trades.empty:
        st.info("No trades on this date.")
    else:
        show_cols = ["Date/Time","Symbol","Side","Quantity","Price","Gross P/L","Execution fee","Net P/L","Description"]
        existing_cols = [c for c in show_cols if c in day_trades.columns]
        for cc in ["Price","Gross P/L","Execution fee","Net P/L"]:
            if cc in day_trades.columns:
                day_trades[cc] = day_trades[cc].apply(lambda x: style_currency(x) if pd.notna(x) else x)
        st.dataframe(day_trades[existing_cols], use_container_width=True, hide_index=True)

with right:
    st.subheader("Summary")
    sel = daily_net[daily_net["date"].dt.date == selected_day]
    day_pnl  = float(sel["pnl"].sum()) if not sel.empty else 0.0
    day_fees = float(sel["fees"].sum()) if not sel.empty else 0.0
    # Month totals (net)
    monthly_pnl  = float(month_df["pnl"].sum()) if not month_df.empty else 0.0
    monthly_fees = float(month_df["fees"].sum()) if not month_df.empty else 0.0
    total_pnl  = float(daily_net["pnl"].sum())
    total_fees = float(daily_net["fees"].sum())
    def metric_row(label, value):
        st.markdown(f"**{label}:** {style_currency(value)}")
    metric_row("Selected Day P&L (Net)", day_pnl)
    metric_row("Selected Day Fees", day_fees)
    st.divider()
    metric_row(f"{current_month.strftime('%B %Y')} P&L (Net)", monthly_pnl)
    metric_row(f"{current_month.strftime('%B %Y')} Fees", monthly_fees)
    st.divider()
    metric_row("Total P&L (all data, Net)", total_pnl)
    metric_row("Total Fees (all data)", total_fees)

st.caption("Tip: Click a day in the calendar. 🟢 profit, 🔴 loss, ⚪ no trades.")

# -----------------------------
# Performance Graph (no 'Daily' option)
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

view = st.selectbox("View", ["Weekly", "Monthly", "Yearly"], index=0)
style = st.radio("Graph style", ["Line + dots", "Bars"], horizontal=True)
basis = st.radio("P&L basis", ["Net (after fees)", "Gross (before fees)"], horizontal=True, index=0)
show_fees = st.checkbox("Show fees in tooltips", value=True)

basis_col = "Net P/L" if basis.startswith("Net") else "Gross P/L"
daily_basis = build_daily(raw, basis_col)

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

# WEEKLY — Sunday→Saturday, tooltip shows winners & losers
if view == "Weekly":
    w_start, w_end = sunday_week_bounds(selected_day)
    rng = pd.date_range(w_start, w_end, freq="D")
    wk = daily_basis.loc[(daily_basis["date"].dt.date >= w_start) & (daily_basis["date"].dt.date <= w_end)][["date","pnl","fees","trades"]].copy()
    wk = wk.set_index("date").reindex(rng, fill_value=0.0).rename_axis("date").reset_index()
    wk["Winners"] = wk["date"].map(day_winners).fillna("—")
    wk["Losers"]  = wk["date"].map(day_losers).fillna("—")
    tooltips = [
        alt.Tooltip("Winners:N", title="Winners"),
        alt.Tooltip("Losers:N", title="Losers"),
        alt.Tooltip("pnl:Q", title="P&L", format="$.2f"),
    ]
    if show_fees and "fees" in wk.columns:
        tooltips.append(alt.Tooltip("fees:Q", title="Fees", format="$.2f"))
    if "trades" in wk.columns:
        tooltips.append(alt.Tooltip("trades:Q", title="Trades"))
    render_chart(wk.rename(columns={"date":"x"}), "x:T", f"Week {w_start.isoformat()} → {w_end.isoformat()}", tooltips)

# MONTHLY — all days in current month
elif view == "Monthly":
    m0, m1 = month_bounds(current_month)
    rng = pd.date_range(m0, m1, freq="D")
    md = daily_basis.loc[(daily_basis["date"].dt.date >= m0) & (daily_basis["date"].dt.date <= m1)][["date","pnl","fees","trades"]].copy()
    md = md.set_index("date").reindex(rng, fill_value=0.0).rename_axis("date").reset_index()
    md["Winners"] = md["date"].map(day_winners).fillna("—")
    md["Losers"]  = md["date"].map(day_losers).fillna("—")
    tooltips = [
        alt.Tooltip("Winners:N", title="Winners"),
        alt.Tooltip("Losers:N", title="Losers"),
        alt.Tooltip("pnl:Q", title="P&L", format="$.2f"),
    ]
    if show_fees and "fees" in md.columns:
        tooltips.append(alt.Tooltip("fees:Q", title="Fees", format="$.2f"))
    if "trades" in md.columns:
        tooltips.append(alt.Tooltip("trades:Q", title="Trades"))
    render_chart(md.rename(columns={"date":"x"}), "x:T", current_month.strftime("Days of %B %Y"), tooltips)

# YEARLY — aggregate by week (Sun→Sat)
elif view == "Yearly":
    year = selected_day.year
    y0 = date(year, 1, 1); y1 = date(year, 12, 31)
    yr = daily_basis.loc[(daily_basis["date"].dt.date >= y0) & (daily_basis["date"].dt.date <= y1)][["date","pnl","fees"]].copy()
    yr = (yr.set_index("date").resample("W-SAT").sum().rename_axis("week_end").reset_index())
    full_weeks = pd.date_range(pd.Timestamp(y0), pd.Timestamp(y1), freq="W-SAT")
    yr = yr.set_index("week_end").reindex(full_weeks, fill_value=0.0).rename_axis("week_end").reset_index()
    yr["week_start"] = yr["week_end"] - pd.Timedelta(days=6)
    yr["Winners"] = yr["week_end"].map(week_winners).fillna("—")
    yr["Losers"]  = yr["week_end"].map(week_losers).fillna("—")
    yr = yr.rename(columns={"week_end":"x"})
    tooltips = [
        alt.Tooltip("Winners:N", title="Winners"),
        alt.Tooltip("Losers:N", title="Losers"),
        alt.Tooltip("pnl:Q", title="P&L", format="$.2f"),
    ]
    if show_fees and "fees" in yr.columns:
        tooltips.append(alt.Tooltip("fees:Q", title="Fees", format="$.2f"))
    render_chart(yr, "x:T", f"Weeks of {year}", tooltips)
