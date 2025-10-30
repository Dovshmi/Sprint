
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
import calendar
import re

# -----------------------------
# Helpers
# -----------------------------
def normalize_numeric(x):
    """Extract numeric from strings like '266.40 USD', '1,234.56', '-2.00', '39 NVDG' -> float or NaN."""
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
    """Reads Colmex 'Filled orders' CSV (semicolon-delimited), parses columns & datetimes (day-first)."""
    df = pd.read_csv(file, sep=";", engine="python")
    # Common numeric fields
    for col in ["Gross P/L", "Execution fee", "Net P/L", "Price", "Quantity"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_numeric)

    if "Date/Time" not in df.columns:
        raise ValueError("Missing 'Date/Time' column in the CSV.")
    df["Date/Time"] = df["Date/Time"].astype(str).str.replace("\xa0", " ").str.strip()
    df["Date/Time"] = df["Date/Time"].str.replace(r"\s+", " ", regex=True)

    # Day-first parsing (e.g., 29.10.2025 15:55:13)
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
    """Return first and last dates (inclusive) for the given month_start (assumed first of month)."""
    last_day = calendar.monthrange(month_start.year, month_start.month)[1]
    start = date(month_start.year, month_start.month, 1)
    end = date(month_start.year, month_start.month, last_day)
    return start, end

def month_weeks_sunday_first(month_start: date):
    """Return list of weeks (each week is list of 7 dates) for the month, Sunday-first."""
    cal = calendar.Calendar(firstweekday=6)  # Sunday = 6
    return cal.monthdatescalendar(month_start.year, month_start.month)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Colmex P&L Calendar", page_icon="📆", layout="wide")
st.title("📆 Colmex Pro — P&L Calendar (Clickable Days)")

with st.expander("How it works"):
    st.markdown(
        """
        Upload your **Colmex Pro Filled orders** CSV (semicolon `;`).  
        Click a **day cell** to see that day's trades.  
        Legend: 🟢 = profit, 🔴 = loss, ⚪ = flat/no trades.
        """
    )

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

# -----------------------------
# Session state for current month & selected day
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

# Slice data for current month
m_start, m_end = month_bounds(st.session_state.current_month)
month_mask = (daily["date"].dt.date >= m_start) & (daily["date"].dt.date <= m_end)
month_df = daily.loc[month_mask].copy()

# Maps
pnl_map = {d.date(): float(p) for d, p in zip(month_df["date"], month_df["pnl"])}
trd_map = {d.date(): int(t) for d, t in zip(month_df["date"], month_df["trades"])}
fee_map = {d.date(): float(f) for d, f in zip(month_df["date"], month_df["fees"])}

# Legend & weekday header
legend_cols = st.columns([1,1,1,7])
with legend_cols[0]: st.markdown("🟢 **Profit**")
with legend_cols[1]: st.markdown("🔴 **Loss**")
with legend_cols[2]: st.markdown("⚪ **Flat**")

WEEKDAYS = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
header_cols = st.columns(7)
for i, name in enumerate(WEEKDAYS):
    with header_cols[i]:
        st.markdown(f"<div style='text-align:center;font-weight:600;'>{name}</div>", unsafe_allow_html=True)

# Calendar grid (buttons per day in current month only)
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
            day_str = f"{d.day:02d} {icon}"
            pnl_str = f"{style_currency(pnl)}"
            label = f"{day_str}\n{pnl_str}"
            help_txt = f"Date: {d.isoformat()} | P&L: {style_currency(pnl)} | Fees: {style_currency(fees)} | Trades: {trades}"
            # Key includes month to avoid duplicates across renders
            if st.button(label, key=f"day_{month_key}_{d.isoformat()}", help=help_txt):
                st.session_state.selected_day = d

# Details + summary
left, right = st.columns([3,2], vertical_alignment="top")

with left:
    st.subheader(f"Trades on {st.session_state.selected_day}")
    day_mask = raw["date"].dt.date == st.session_state.selected_day
    day_trades = raw.loc[day_mask].copy()
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
    sel = daily[daily["date"].dt.date == st.session_state.selected_day]
    day_pnl = float(sel["pnl"].sum()) if not sel.empty else 0.0
    day_fees = float(sel["fees"].sum()) if not sel.empty else 0.0

    monthly_pnl = float(month_df["pnl"].sum()) if not month_df.empty else 0.0
    monthly_fees = float(month_df["fees"].sum()) if not month_df.empty else 0.0
    total_pnl = float(daily["pnl"].sum())
    total_fees = float(daily["fees"].sum())

    def metric_row(label, value):
        st.markdown(f"**{label}:** {style_currency(value)}")

    metric_row("Selected Day P&L", day_pnl)
    metric_row("Selected Day Fees", day_fees)
    st.divider()
    metric_row(f"{current_month.strftime('%B %Y')} P&L", monthly_pnl)
    metric_row(f"{current_month.strftime('%B %Y')} Fees", monthly_fees)
    st.divider()
    metric_row("Total P&L (all data)", total_pnl)
    metric_row("Total Fees (all data)", total_fees)

st.caption("Tip: Click a day in the calendar above. 🟢 profit, 🔴 loss, ⚪ no trades.")
