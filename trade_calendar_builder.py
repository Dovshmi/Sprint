#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Trade Calendar — PRO (Dark+Blue UI, clearer calendar, slimmer sidepane)

What’s new vs your last file:
  • True dark theme (deep navy) + modern blue buttons
  • “6 trades” (or “1 trade”) instead of “#6”
  • Better contrast & slightly larger fonts in the calendar cells
  • Right side pane narrowed for more calendar room

CSV expectation (semicolon-separated preferred):
  "Date/Time" (dd/mm/YYYY HH:MM:SS), "Net P/L"
Optional:
  "Symbol","Side","Quantity","Price","Trading exchange"

Dependencies:
  pip install pandas pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import date, datetime
import calendar
import re
import json
import os

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ----------------------------- Constants & Paths -----------------------------
APP_DIR = Path(os.path.expanduser("~")) / ".trade_calendar"
APP_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = APP_DIR / "config.json"
NOTES_PATH = APP_DIR / "notes.json"

DEFAULT_CFG = {
    "theme": "dark",           # "dark" or "light"
    "firstweekday": "SUN",     # "SUN" or "MON"
    "profits_only": False,
    "symbol_filter": "",
    "last_csvs": []            # list of recently used CSV paths
}

# UI sizes
SIDEPANE_W = 340  # slimmer side pane

# --- THEME PALETTE (Dark) ---
DARK_BG          = "#0b1020"   # app background
DARK_SURFACE     = "#111827"   # frames / panes
DARK_SURFACE_2   = "#0f172a"   # canvas / alt
DARK_TEXT_MAIN   = "#e5e7eb"
DARK_TEXT_DIM    = "#9aa3b2"
DARK_DIVIDER     = "#2a2f3a"

ACCENT_BLUE      = "#2563eb"   # primary buttons
ACCENT_BLUE_HOV  = "#1d4ed8"
ACCENT_BLUE_ACT  = "#1e40af"
ACCENT_BLUE_TXT  = "#ffffff"

POS_GREEN        = "#22c55e"
NEG_RED          = "#ef4444"
NEUTRAL_GRAY     = "#94a3b8"

# ----------------------------- Utilities -----------------------------
def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return {**DEFAULT_CFG, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return DEFAULT_CFG.copy()

def save_config(cfg: dict):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def load_notes() -> dict:
    if NOTES_PATH.exists():
        try:
            return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_notes(notes: dict):
    try:
        NOTES_PATH.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def money_to_float(text):
    if pd.isna(text):
        return np.nan
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip()
    sign = -1.0 if s.startswith('-') else 1.0
    m = re.search(r'\d+(?:\.\d+)?', s)
    return sign * float(m.group(0)) if m else np.nan

def fmt_usd(v, signed=True):
    if v is None or pd.isna(v):
        return "-"
    if signed:
        sign = "+" if v > 0 else ("-" if v < 0 else "")
        return f"{sign}${abs(float(v)):,.2f}"
    return f"${float(v):,.2f}"

def try_read_csv(path: Path) -> pd.DataFrame:
    # prefer semicolon; fall back to comma
    try:
        df = pd.read_csv(path, sep=';')
    except Exception:
        df = pd.read_csv(path)
    return df

def normalize_broker_csv(df: pd.DataFrame) -> pd.DataFrame:
    if 'Date/Time' not in df.columns or 'Net P/L' not in df.columns:
        raise ValueError("CSV must include 'Date/Time' and 'Net P/L' columns.")
    # parse datetime
    try:
        df['Date'] = pd.to_datetime(df['Date/Time'], format='%d/%m/%Y %H:%M:%S')
    except Exception:
        df['Date'] = pd.to_datetime(df['Date/Time'], errors='coerce')
    if df['Date'].isna().all():
        raise ValueError("Could not parse 'Date/Time'. Expected like '24/10/2025 16:31:42'.")
    # parse net pnl
    df['Net_PnL'] = df['Net P/L'].apply(money_to_float)
    # optional
    for col in ['Symbol','Side','Quantity','Price','Trading exchange']:
        if col not in df.columns:
            df[col] = None
    return df

def merge_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = normalize_broker_csv(try_read_csv(p))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    # dedupe heuristic: Date/Time + Symbol + Side + Quantity + Price
    dedupe_cols = [c for c in ['Date/Time','Symbol','Side','Quantity','Price'] if c in all_df.columns]
    if dedupe_cols:
        all_df = all_df.drop_duplicates(subset=dedupe_cols, keep='first').reset_index(drop=True)
    return all_df.sort_values('Date').reset_index(drop=True)

# ----------------------------- Aggregations -----------------------------
def daily_summary(df: pd.DataFrame, symbol_filter: str|None=None) -> pd.DataFrame:
    d = df.copy()
    if symbol_filter:
        d = d[d['Symbol'].astype(str).str.upper().str.contains(symbol_filter.upper(), na=False)]
    daily = (d.groupby(d['Date'].dt.date)
               .agg(net=('Net_PnL','sum'),
                    orders=('Date','count'))
               .reset_index()
               .rename(columns={'Date':'day'})
               .sort_values('day'))
    return daily

def per_day_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['day'] = d['Date'].dt.date
    by = (d.groupby(['day','Symbol'])
            .agg(net=('Net_PnL','sum'),
                 orders=('Date','count'))
            .reset_index()
            .sort_values(['day','net'], ascending=[True, False]))
    return by

def trades_by_day(df: pd.DataFrame) -> dict:
    d = df.copy()
    d['day'] = d['Date'].dt.date
    return {g: gdf.sort_values('Date').reset_index(drop=True) for g, gdf in d.groupby('day')}

def monthly_slice(df: pd.DataFrame, year: int, month: int, symbol_filter: str|None=None) -> pd.DataFrame:
    d = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month)]
    if symbol_filter:
        d = d[d['Symbol'].astype(str).str.upper().str.contains(symbol_filter.upper(), na=False)]
    return d

def monthly_stats(df: pd.DataFrame, year: int, month: int, symbol_filter: str|None=None) -> dict:
    d = monthly_slice(df, year, month, symbol_filter)
    if d.empty:
        return dict(net=0.0, orders=0, trade_days=0, pos_days=0, pos_rate=0.0, avg_per_order=0.0)
    dd = d.groupby(d['Date'].dt.date)['Net_PnL'].sum()
    pos_days = int((dd > 0).sum())
    trade_days = int((dd != 0).sum()) if len(dd) else 0
    orders = int(len(d))
    net = float(d['Net_PnL'].sum())
    avg_per_order = float(net / orders) if orders else 0.0
    pos_rate = float(100 * pos_days / max(trade_days, 1))
    return dict(net=net, orders=orders, trade_days=trade_days, pos_days=pos_days,
                pos_rate=pos_rate, avg_per_order=avg_per_order)

# ----------------------------- PNG render (offscreen) -----------------------------
def draw_month_to_image(year:int, month:int, day_data:dict, firstweekday:int, profits_only:bool, symbol_filter:str|None, theme:str) -> Image.Image:
    # theme colors
    if theme == "light":
        bg = (247,247,249); fg=(24,24,28); grid=(210,210,215); green=(25,128,70); red=(200,60,60); gray=(110,110,120)
    else:
        # align with real dark theme
        bg = (11,16,32); fg=(229,231,235); grid=(42,47,58); green=(34,197,94); red=(239,68,68); gray=(148,163,184)

    W, H = 1400, 900
    margin = 20
    header_h = 90
    cell_h = (H - header_h - 2*margin) // 6
    cell_w = (W - 2*margin) // 7

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("arial.ttf", 28)
        font_mid = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font_big = ImageFont.load_default()
        font_mid = ImageFont.load_default()
        font_small = ImageFont.load_default()

    title = f"{calendar.month_name[month]} {year} — {'Profits' if profits_only else 'P&L'}"
    if symbol_filter:
        title += f"  (filter: {symbol_filter})"
    draw.text((margin, margin), title, fill=fg, font=font_big)

    labels = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'] if firstweekday == 6 else ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for i, lab in enumerate(labels):
        x = margin + i*cell_w + cell_w//2
        draw.text((x-20, margin+40), lab, fill=fg, font=font_mid)

    cal = calendar.Calendar(firstweekday=firstweekday)
    month_days = list(cal.itermonthdates(year, month))
    for r in range(6):
        for c in range(7):
            idx = r*7 + c
            if idx >= len(month_days): continue
            d = month_days[idx]
            x0 = margin + c*cell_w
            y0 = margin + header_h + r*cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            draw.rectangle([x0, y0, x1, y1], outline=grid, width=1)

            in_month = (d.month == month)
            dc = fg if in_month else gray
            draw.text((x0+8, y0+6), str(d.day), fill=dc, font=font_small)

            info = day_data.get(d, None)
            if in_month and info:
                net = info['net']
                txt = "-" if (profits_only and (net <= 0)) else fmt_usd(net)
                color = green if net > 0 else (red if net < 0 else gray)
                bb = draw.textbbox((0,0), txt, font=font_mid)
                tw, th = bb[2]-bb[0], bb[3]-bb[1]
                draw.text((x0 + (cell_w - tw)//2, y0 + (cell_h - th)//2 - 8), txt, fill=color, font=font_mid)
                # orders corner -> "n trades"
                ords = int(info.get('orders', 0))
                otext = f"{ords} trade" if ords == 1 else f"{ords} trades"
                bb2 = draw.textbbox((0,0), otext, font=font_small)
                draw.text((x1 - (bb2[2]-bb2[0]) - 8, y1 - (bb2[3]-bb2[1]) - 8), otext, fill=dc, font=font_small)
                # top symbol line
                top = info.get('top', None)
                if top:
                    bb3 = draw.textbbox((0,0), top, font=font_small)
                    draw.text((x0 + (cell_w - (bb3[2]-bb3[0]))//2, y0 + (cell_h - th)//2 + 16),
                              top, fill=dc, font=font_small)
    return img

# ----------------------------- ICS export -----------------------------
def export_ics(df: pd.DataFrame, out: Path):
    def to_ics_dt(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y%m%dT%H%M%S")
    lines = ["BEGIN:VCALENDAR","VERSION:2.0","PRODID:-//TradeCalendarPRO//EN"]
    for i, r in df.iterrows():
        ts = pd.to_datetime(r['Date'])
        start = to_ics_dt(ts)
        end   = to_ics_dt(ts + pd.Timedelta(minutes=5))
        symbol = str(r.get('Symbol') or "")
        side   = str(r.get('Side') or "")
        qty    = int(r.get('Quantity')) if ('Quantity' in r and not pd.isna(r['Quantity'])) else 0
        price  = str(r.get('Price') or "")
        net    = str(r.get('Net P/L') or "")
        exch   = str(r.get('Trading exchange') or "")
        title = " ".join([x for x in [side, str(qty) if qty else "", symbol, f"@ {price}" if price else ""] if x])
        if not title: title = f"Trade {i+1}"
        desc = "\\n".join([p for p in [
            f"Symbol: {symbol}" if symbol else "",
            f"Side: {side}" if side else "",
            f"Qty: {qty}" if qty else "",
            f"Price: {price}" if price else "",
            f"Net P/L: {net}" if net else "",
            f"Exchange: {exch}" if exch else ""
        ] if p])
        uid = f"{start}-{symbol}-{i}@tradecalendarpro"
        lines += ["BEGIN:VEVENT", f"UID:{uid}", f"DTSTAMP:{start}", f"DTSTART:{start}", f"DTEND:{end}",
                  f"SUMMARY:{title}", f"DESCRIPTION:{desc}", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    out.write_text("\n".join(lines), encoding="utf-8")

# ----------------------------- App -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.notes = load_notes()

        self.title("Interactive Trade Calendar — PRO")
        self.geometry("1300x900")

        # data holders
        self.df = pd.DataFrame()
        self.by_symbol = pd.DataFrame()
        self.trades_map = {}
        self.symbol_filter = tk.StringVar(value=self.cfg.get("symbol_filter",""))
        self.profits_only = tk.BooleanVar(value=bool(self.cfg.get("profits_only", False)))
        self.firstweekday = tk.StringVar(value=self.cfg.get("firstweekday","SUN"))
        today = date.today()
        self.sel_year = tk.IntVar(value=today.year)
        self.sel_month = tk.IntVar(value=today.month)
        self.selected_day = None

        # swipe
        self.swipe_start = None

        self.style = ttk.Style(self)
        self.apply_theme(self.cfg.get("theme","dark"))

        self.build_menu()
        self.build_ui()
        self.bind("<Left>", lambda e: self.change_month(-1))
        self.bind("<Right>", lambda e: self.change_month(+1))

        # Autoload last CSVs if available
        last_csvs = [Path(p) for p in self.cfg.get("last_csvs", []) if Path(p).exists()]
        if last_csvs:
            self.load_csvs(last_csvs)

    # ---------- Menus / Theme ----------
    def build_menu(self):
        m = tk.Menu(self)
        self.config(menu=m)

        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label="Open CSV…", command=self.open_csv)
        fm.add_command(label="Open Multiple CSVs…", command=self.open_multiple_csvs)
        fm.add_separator()
        fm.add_command(label="Export Month PNG…", command=self.export_png)
        fm.add_command(label="Export ICS…", command=self.export_ics_all)
        fm.add_separator()
        fm.add_command(label="Save Settings", command=self.save_settings)
        fm.add_command(label="Exit", command=self.destroy)
        m.add_cascade(label="File", menu=fm)

        vm = tk.Menu(m, tearoff=0)
        vm.add_command(label="Dark theme", command=lambda: self.set_theme("dark"))
        vm.add_command(label="Light theme", command=lambda: self.set_theme("light"))
        m.add_cascade(label="View", menu=vm)

        tm = tk.Menu(m, tearoff=0)
        tm.add_command(label="Equity curve (All data)", command=lambda: self.show_equity_curve(month_only=False))
        tm.add_command(label="Equity curve (Current month)", command=lambda: self.show_equity_curve(month_only=True))
        tm.add_separator()
        tm.add_command(label="Reset filters", command=self.reset_filters)
        m.add_cascade(label="Tools", menu=tm)

        hm = tk.Menu(m, tearoff=0)
        hm.add_command(label="Shortcuts / Help", command=self.show_help)
        m.add_cascade(label="Help", menu=hm)

    def apply_theme(self, theme: str):
        # Modern theming
        self.style.theme_use('clam')
        if theme == "light":
            # minimal light mapping
            self.configure(bg="white")
            self.style.configure('.', background="white", foreground="black")
            self.style.configure('TFrame', background="white")
            self.style.configure('TLabel', background="white", foreground="black")
            self.style.configure('TCheckbutton', background="white", foreground="black")
            self.style.configure('TEntry', fieldbackground="white", foreground="black")
            self.style.configure('Treeview',
                                 background="white",
                                 fieldbackground="white",
                                 foreground="black",
                                 bordercolor="#d1d5db")
            self.style.map('Treeview', background=[('selected', '#e5e7eb')], foreground=[('selected', 'black')])
            self.style.configure('TButton', padding=8)
        else:
            # REAL dark + blue buttons
            self.configure(bg=DARK_BG)
            self.style.configure('.', background=DARK_SURFACE, foreground=DARK_TEXT_MAIN)
            self.style.configure('TFrame', background=DARK_SURFACE)
            self.style.configure('TLabel', background=DARK_SURFACE, foreground=DARK_TEXT_MAIN)
            self.style.configure('TCheckbutton', background=DARK_SURFACE, foreground=DARK_TEXT_MAIN)
            self.style.configure('TEntry', fieldbackground=DARK_SURFACE_2, foreground=DARK_TEXT_MAIN)
            self.style.configure('Treeview',
                                 background=DARK_SURFACE_2,
                                 fieldbackground=DARK_SURFACE_2,
                                 foreground=DARK_TEXT_MAIN,
                                 bordercolor=DARK_DIVIDER)
            self.style.map('Treeview',
                           background=[('selected', '#1f2937')],
                           foreground=[('selected', '#f3f4f6')])

            # Blue buttons
            self.style.configure('TButton',
                                 background=ACCENT_BLUE,
                                 foreground=ACCENT_BLUE_TXT,
                                 padding=10,
                                 borderwidth=0,
                                 focusthickness=0)
            self.style.map('TButton',
                           background=[('active', ACCENT_BLUE_ACT),
                                       ('pressed', ACCENT_BLUE_HOV),
                                       ('disabled', '#334155')],
                           foreground=[('disabled', '#94a3b8')])

        self.cfg["theme"] = theme

    def set_theme(self, theme: str):
        self.apply_theme(theme)
        # update canvas/notes bg to match theme
        if hasattr(self, "canvas"):
            self.canvas.configure(bg=DARK_SURFACE_2 if self.cfg.get("theme")=="dark" else "#ffffff")
        if hasattr(self, "notes_text"):
            self.notes_text.configure(
                bg=(DARK_SURFACE_2 if self.cfg.get("theme")=="dark" else "white"),
                fg=(DARK_TEXT_MAIN if self.cfg.get("theme")=="dark" else "black"),
                insertbackground=(DARK_TEXT_MAIN if self.cfg.get("theme")=="dark" else "black")
            )
        self.refresh_calendar()

    # ---------- UI layout ----------
    def build_ui(self):
        # Header
        hdr = ttk.Frame(self)
        hdr.pack(side=tk.TOP, fill=tk.X, padx=12, pady=8)

        left = ttk.Frame(hdr)
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.total_label = ttk.Label(left, text="Total Net P&L: –", font=("Segoe UI", 12, "bold"))
        self.total_label.pack(anchor="w")

        self.month_label_stats = ttk.Label(left, text="Month P&L: –  |  Trading Days: –  |  Positive Day Rate: –  |  Avg/Order: –",
                                           font=("Segoe UI", 10))
        self.month_label_stats.pack(anchor="w", pady=(2,0))

        right = ttk.Frame(hdr)
        right.pack(side=tk.RIGHT)
        ttk.Label(right, text="Symbol filter:").grid(row=0, column=0, sticky="e")
        ttk.Entry(right, textvariable=self.symbol_filter, width=14).grid(row=0, column=1, padx=4)
        ttk.Checkbutton(right, text="Profits only", variable=self.profits_only, command=self.refresh_calendar).grid(row=0, column=2, padx=6)
        ttk.Label(right, text="First weekday:").grid(row=0, column=3, sticky="e")
        ttk.OptionMenu(right, self.firstweekday, self.firstweekday.get(), "SUN", "MON",
                       command=lambda *_: self.refresh_calendar()).grid(row=0, column=4)

        ttk.Label(right, text="Month:").grid(row=1, column=0, sticky="e")
        ttk.Spinbox(right, from_=1, to=12, textvariable=self.sel_month, width=4, command=self.refresh_calendar).grid(row=1, column=1)
        ttk.Label(right, text="Year:").grid(row=1, column=2, sticky="e")
        ttk.Spinbox(right, from_=1990, to=2099, textvariable=self.sel_year, width=6, command=self.refresh_calendar).grid(row=1, column=3)
        ttk.Button(right, text="Prev", command=lambda: self.change_month(-1)).grid(row=1, column=4, padx=4)
        ttk.Button(right, text="Next", command=lambda: self.change_month(+1)).grid(row=1, column=5, padx=4)

        # Body
        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Calendar canvas (swipeable)
        self.canvas = tk.Canvas(
            body,
            bg=DARK_SURFACE_2 if self.cfg.get("theme")=="dark" else "#ffffff",
            height=620,
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,8))
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Button-1>", self.on_click)

        # Right pane with details + notes (slimmer)
        rightpane = ttk.Frame(body, width=SIDEPANE_W)
        rightpane.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.day_label = ttk.Label(rightpane, text="(Select a day)", font=("Segoe UI", 12, "bold"))
        self.day_label.pack(anchor="w", pady=(4,6))

        ttk.Label(rightpane, text="Per-stock P&L").pack(anchor="w")
        self.tree_sym = ttk.Treeview(rightpane, columns=("symbol","net","orders"), show="headings", height=6)
        self.tree_sym.heading("symbol", text="Symbol")
        self.tree_sym.heading("net", text="Net P&L")
        self.tree_sym.heading("orders", text="#")
        self.tree_sym.column("symbol", width=110, anchor="w")
        self.tree_sym.column("net", width=110, anchor="e")
        self.tree_sym.column("orders", width=40, anchor="center")
        self.tree_sym.pack(fill=tk.X, pady=(2,8))

        ttk.Label(rightpane, text="Trades").pack(anchor="w")
        self.tree_tr = ttk.Treeview(rightpane, columns=("time","side","qty","symbol","price","net"), show="headings", height=9)
        for col, txt, w, anc in [
            ("time","Time",100,"w"),
            ("side","Side",50,"center"),
            ("qty","Qty",50,"e"),
            ("symbol","Symbol",70,"w"),
            ("price","Price",80,"e"),
            ("net","Net P&L",90,"e"),
        ]:
            self.tree_tr.heading(col, text=txt)
            self.tree_tr.column(col, width=w, anchor=anc)
        self.tree_tr.pack(fill=tk.BOTH, expand=True, pady=(2,6))

        # Notes
        notes_frame = ttk.LabelFrame(rightpane, text="Day Notes")
        notes_frame.pack(fill=tk.X, pady=(6,6))
        self.notes_text = tk.Text(
            notes_frame, height=3, wrap="word",
            bg=(DARK_SURFACE_2 if self.cfg.get("theme")=="dark" else "white"),
            fg=(DARK_TEXT_MAIN if self.cfg.get("theme")=="dark" else "black"),
            insertbackground=(DARK_TEXT_MAIN if self.cfg.get("theme")=="dark" else "black"),
            relief="flat"
        )
        self.notes_text.pack(fill=tk.X, padx=6, pady=6)
        btns = ttk.Frame(notes_frame)
        btns.pack(fill=tk.X, padx=6, pady=(0,6))
        ttk.Button(btns, text="Save Note", command=self.save_day_note).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear", command=lambda: self.notes_text.delete("1.0","end")).pack(side=tk.LEFT, padx=4)

        # Bottom exports / charts
        export = ttk.Frame(rightpane)
        export.pack(fill=tk.X, pady=(6,0))
        ttk.Button(export, text="Export Selected Day CSV", command=self.export_day_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(export, text="Equity Curve (All)", command=lambda: self.show_equity_curve(False)).pack(side=tk.LEFT, padx=4)
        ttk.Button(export, text="Equity Curve (Month)", command=lambda: self.show_equity_curve(True)).pack(side=tk.LEFT, padx=4)

        self.cell_rects = {}
        self.refresh_calendar()

    # ---------- CSV open/merge ----------
    def open_csv(self):
        p = filedialog.askopenfilename(title="Choose CSV", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not p:
            return
        self.load_csvs([Path(p)])

    def open_multiple_csvs(self):
        ps = filedialog.askopenfilenames(title="Choose CSVs", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not ps:
            return
        paths = [Path(x) for x in ps]
        self.load_csvs(paths)

    def load_csvs(self, paths: list[Path]):
        try:
            self.df = merge_csvs(paths)
            if self.df.empty:
                messagebox.showinfo("No data", "No trades found.")
                return
            self.by_symbol = per_day_by_symbol(self.df)
            self.trades_map = trades_by_day(self.df)

            # all-time total
            total_net = float(self.df['Net_PnL'].sum())
            self.total_label.config(text=f"Total Net P&L: {fmt_usd(total_net)}")

            # jump to last trade month
            last = self.df['Date'].max().date()
            self.sel_year.set(last.year)
            self.sel_month.set(last.month)

            # remember CSVs
            self.cfg["last_csvs"] = [str(p) for p in paths]
            save_config(self.cfg)

            self.refresh_calendar()
        except Exception as e:
            messagebox.showerror("Failed to load CSV", str(e))

    # ---------- Calendar / drawing ----------
    def refresh_calendar(self):
        self.canvas.delete("all")
        self.cell_rects.clear()
        if self.df.empty:
            self.canvas.create_text(20, 40, anchor="w",
                                    fill=(DARK_TEXT_MAIN if self.cfg.get("theme")=="dark" else "#111"),
                                    font=("Segoe UI", 14, "bold"),
                                    text="Open a CSV to view your trade calendar.")
            self.month_label_stats.config(text="Month P&L: –  |  Trading Days: –  |  Positive Day Rate: –  |  Avg/Order: –")
            return

        sym_filter = self.symbol_filter.get().strip() or None
        year, month = int(self.sel_year.get()), int(self.sel_month.get())
        fw = 6 if self.firstweekday.get() == "SUN" else 0

        # monthly stats
        mstats = monthly_stats(self.df, year, month, sym_filter)
        self.month_label_stats.config(
            text=f"Month P&L: {fmt_usd(mstats['net'])}  |  "
                 f"Trading Days: {mstats['trade_days']}  |  "
                 f"Positive Day Rate: {mstats['pos_rate']:.0f}%  |  "
                 f"Avg/Order: {fmt_usd(mstats['avg_per_order'])}"
        )

        # daily data
        daily = daily_summary(self.df, sym_filter)
        day_map = {r['day']: {'net': float(r['net']), 'orders': int(r['orders'])} for _, r in daily.iterrows()}

        # top symbol line per day
        by = self.by_symbol
        if sym_filter:
            by = by[by['Symbol'].astype(str).str.upper().str.contains(sym_filter.upper(), na=False)]
        for d, g in by.groupby('day'):
            if d in day_map and not g.empty:
                top_row = g.sort_values('net', ascending=False).iloc[0]
                day_map[d]['top'] = f"{top_row['Symbol']} {fmt_usd(top_row['net'])}"

        self.draw_month_grid(year, month, fw, day_map, self.profits_only.get(), sym_filter)

        # update config with current toggles
        self.cfg["profits_only"] = bool(self.profits_only.get())
        self.cfg["firstweekday"] = self.firstweekday.get()
        self.cfg["symbol_filter"] = self.symbol_filter.get()
        save_config(self.cfg)

    def draw_month_grid(self, year:int, month:int, firstweekday:int, day_data:dict, profits_only:bool, sym_filter:str|None):
        # color depends on theme
        theme = self.cfg.get("theme","dark")
        title_color     = ("#111" if theme=="light" else DARK_TEXT_MAIN)
        head_color      = ("#333" if theme=="light" else DARK_TEXT_DIM)
        date_color_main = ("#111" if theme=="light" else DARK_TEXT_MAIN)
        date_color_dim  = ("#777" if theme=="light" else "#64748b")

        W = self.canvas.winfo_width() or 1200
        H = self.canvas.winfo_height() or 620
        margin = 16
        title_h = 40
        header_h = 26
        rows = 6
        cols = 7
        cell_h = (H - title_h - header_h - 2*margin) // rows
        cell_w = (W - 2*margin) // cols

        title = f"{calendar.month_name[month]} {year} — {'Profits' if profits_only else 'P&L'}"
        if sym_filter:
            title += f"  (filter: {sym_filter})"
        self.canvas.create_text(margin, margin, anchor="nw", fill=title_color, font=("Segoe UI", 14, "bold"), text=title)

        labels = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'] if firstweekday == 6 else ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        for c, lab in enumerate(labels):
            x = margin + c*cell_w + cell_w//2
            y = margin + title_h
            self.canvas.create_text(x, y, anchor="n", fill=head_color, font=("Segoe UI", 11, "bold"), text=lab)

        cal = calendar.Calendar(firstweekday=firstweekday)
        month_days = list(cal.itermonthdates(year, month))

        y0 = margin + title_h + header_h
        for r in range(6):
            for c in range(7):
                idx = r*7 + c
                if idx >= len(month_days): continue
                d = month_days[idx]
                x = margin + c*cell_w
                y = y0 + r*cell_h
                x2, y2 = x + cell_w - 2, y + cell_h - 2

                outline = "#cfd3d8" if theme=="light" else DARK_DIVIDER
                self.canvas.create_rectangle(x, y, x2, y2, outline=outline)

                in_month = (d.month == month)
                dc = date_color_main if in_month else date_color_dim
                # bigger date number for readability
                self.canvas.create_text(x+6, y+4, anchor="nw", fill=dc, font=("Segoe UI", 10, "bold"), text=str(d.day))

                info = day_data.get(d, None)
                if in_month and info:
                    net = info['net']
                    txt = "-" if (profits_only and (net <= 0)) else fmt_usd(net)
                    color = POS_GREEN if net > 0 else (NEG_RED if net < 0 else NEUTRAL_GRAY)
                    # slightly larger center value
                    self.canvas.create_text(x + cell_w/2, y + cell_h/2 - 6, anchor="center",
                                            fill=color, font=("Segoe UI", 12, "bold"), text=txt)
                    # "n trades" instead of "#n"
                    orders = int(info.get('orders', 0))
                    orders_txt = f"{orders} trade" if orders == 1 else f"{orders} trades"
                    self.canvas.create_text(x2-8, y2-8, anchor="se",
                                            fill=DARK_TEXT_DIM if theme=="dark" else "#666",
                                            font=("Segoe UI", 9), text=orders_txt)
                    # top symbol line
                    if 'top' in info and info['top']:
                        self.canvas.create_text(x + cell_w/2, y + cell_h/2 + 12, anchor="center",
                                                fill=dc, font=("Segoe UI", 9), text=info['top'])

                self.cell_rects[(x, y, x2, y2)] = d

    # ---------- Notes ----------
    def save_day_note(self):
        if not self.selected_day:
            messagebox.showinfo("Info", "Select a day first.")
            return
        key = self.selected_day.strftime("%Y-%m-%d")
        self.notes[key] = self.notes_text.get("1.0","end").strip()
        save_notes(self.notes)
        messagebox.showinfo("Saved", "Note saved.")

    def load_day_note(self, d: date):
        key = d.strftime("%Y-%m-%d")
        self.notes_text.delete("1.0","end")
        if key in self.notes and self.notes[key]:
            self.notes_text.insert("1.0", self.notes[key])

    # ---------- Interactions (swipe + click) ----------
    def on_click(self, event):
        if self.swipe_start:
            x0, y0, t0 = self.swipe_start
            if abs(event.x - x0) > 10 or abs(event.y - y0) > 10:
                return
        self.select_day_at(event.x, event.y)

    def select_day_at(self, ex, ey):
        if self.df.empty: return
        for (x0,y0,x1,y1), dt in self.cell_rects.items():
            if x0 <= ex <= x1 and y0 <= ey <= y1:
                if dt.year == int(self.sel_year.get()) and dt.month == int(self.sel_month.get()):
                    self.show_day(dt)
                else:
                    self.show_day(None)
                return

    def on_press(self, event):
        self.swipe_start = (event.x, event.y, datetime.now())

    def on_drag(self, event):
        pass

    def on_release(self, event):
        if not self.swipe_start:
            return
        x0, y0, t0 = self.swipe_start
        dx = event.x - x0
        dy = event.y - y0
        dt = (datetime.now() - t0).total_seconds()
        self.swipe_start = None
        horizontal = abs(dx) > 60 and abs(dy) < 80
        quick = dt < 0.35 and abs(dx) > 40
        if horizontal or quick:
            if dx < 0:   # left -> next month
                self.change_month(+1)
            elif dx > 0: # right -> previous month
                self.change_month(-1)

    def change_month(self, delta):
        y, m = int(self.sel_year.get()), int(self.sel_month.get())
        m += delta
        if m <= 0:
            m = 12; y -= 1
        elif m > 12:
            m = 1; y += 1
        self.sel_year.set(y)
        self.sel_month.set(m)
        self.refresh_calendar()

    def show_day(self, d: date|None):
        self.selected_day = d
        # clear tables
        for t in (self.tree_sym, self.tree_tr):
            for i in t.get_children(): t.delete(i)

        if d is None:
            self.day_label.config(text="(Select a day)")
            self.notes_text.delete("1.0","end")
            return

        self.day_label.config(text=d.strftime("%A, %d %B %Y"))

        # per-stock for that day
        if not self.by_symbol.empty:
            sym_filter = self.symbol_filter.get().strip()
            data = self.by_symbol[self.by_symbol['day'] == d]
            if sym_filter:
                data = data[data['Symbol'].astype(str).str.upper().str.contains(sym_filter.upper(), na=False)]
            data = data.sort_values('net', ascending=False)
            for _, r in data.iterrows():
                self.tree_sym.insert("", "end", values=(r['Symbol'], fmt_usd(r['net']), int(r['orders'])))

        # trades for that day
        tdf = self.trades_map.get(d, pd.DataFrame())
        if not tdf.empty:
            sym_filter = self.symbol_filter.get().strip()
            if sym_filter:
                tdf = tdf[tdf['Symbol'].astype(str).str.upper().str.contains(sym_filter.upper(), na=False)]
            for _, r in tdf.iterrows():
                self.tree_tr.insert("", "end", values=(
                    r['Date'].strftime("%H:%M:%S"),
                    r['Side'] or "",
                    int(r['Quantity']) if not pd.isna(r['Quantity']) else "",
                    r['Symbol'] or "",
                    str(r['Price']) if r['Price'] is not None else "",
                    fmt_usd(r['Net_PnL'])
                ))

        # load note
        self.load_day_note(d)

    # ---------- Exports / Charts ----------
    def export_png(self):
        if self.df.empty:
            messagebox.showinfo("Info", "Load a CSV first."); return
        year, month = int(self.sel_year.get()), int(self.sel_month.get())
        fw = 6 if self.firstweekday.get() == "SUN" else 0
        sym_filter = self.symbol_filter.get().strip() or None

        daily = daily_summary(self.df, sym_filter)
        day_map = {r['day']: {'net': float(r['net']), 'orders': int(r['orders'])} for _, r in daily.iterrows()}

        by = self.by_symbol
        if sym_filter:
            by = by[by['Symbol'].astype(str).str.upper().str.contains(sym_filter.upper(), na=False)]
        for d, g in by.groupby('day'):
            if d in day_map and not g.empty:
                top_row = g.sort_values('net', ascending=False).iloc[0]
                day_map[d]['top'] = f"{top_row['Symbol']} {fmt_usd(top_row['net'])}"

        img = draw_month_to_image(year, month, day_map, fw, self.profits_only.get(), sym_filter, self.cfg.get("theme","dark"))

        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")], title="Save calendar PNG")
        if not p: return
        try:
            img.save(p, format="PNG")
            messagebox.showinfo("Saved", f"PNG saved to:\n{p}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def export_ics_all(self):
        if self.df.empty:
            messagebox.showinfo("Info", "Load a CSV first."); return
        p = filedialog.asksaveasfilename(defaultextension=".ics", filetypes=[("iCalendar","*.ics")], title="Save ICS")
        if not p: return
        try:
            export_ics(self.df, Path(p))
            messagebox.showinfo("Saved", f"ICS saved to:\n{p}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def export_day_csv(self):
        if not self.selected_day:
            messagebox.showinfo("Info", "Select a day first."); return
        tdf = self.trades_map.get(self.selected_day, pd.DataFrame())
        if tdf.empty:
            messagebox.showinfo("Info", "No trades on that day."); return
        sym_filter = self.symbol_filter.get().strip()
        if sym_filter:
            tdf = tdf[tdf['Symbol'].astype(str).str.upper().str.contains(sym_filter.upper(), na=False)]
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")], title="Save day CSV")
        if not p: return
        try:
            cols = [c for c in ["Date/Time","Symbol","Side","Quantity","Price","Net P/L","Trading exchange"] if c in tdf.columns]
            tdf[cols].to_csv(p, index=False)
            messagebox.showinfo("Saved", f"CSV saved to:\n{p}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def show_equity_curve(self, month_only: bool):
        if self.df.empty:
            messagebox.showinfo("Info", "Load a CSV first."); return

        year, month = int(self.sel_year.get()), int(self.sel_month.get())
        sym_filter = self.symbol_filter.get().strip() or None
        if month_only:
            d = monthly_slice(self.df, year, month, sym_filter)
        else:
            d = self.df.copy()
            if sym_filter:
                d = d[d['Symbol'].astype(str).str.upper().str.contains(sym_filter.upper(), na=False)]
        if d.empty:
            messagebox.showinfo("Info", "No trades for the selected scope."); return

        dx = d.sort_values('Date')
        dx['cum'] = dx['Net_PnL'].cumsum()

        # Draw simple line chart with PIL (no matplotlib dependency)
        W, H = 900, 360
        margin = 50
        bg = DARK_SURFACE_2 if self.cfg.get("theme")=="dark" else "#ffffff"
        fg = DARK_TEXT_MAIN if self.cfg.get("theme")=="dark" else "#222222"
        grid = DARK_DIVIDER if self.cfg.get("theme")=="dark" else "#e6e9ef"
        line = "#4da3ff"
        img = Image.new("RGB", (W, H), bg)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
            font_b = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
            font_b = ImageFont.load_default()

        # axes
        x0, y0 = margin, H - margin
        x1, y1 = W - margin, margin
        draw.rectangle([x0, y1, x1, y0], outline=grid, width=1)

        ys = dx['cum'].values.astype(float)
        xs = np.linspace(x0+10, x1-10, len(ys))
        ymin, ymax = float(np.min(ys)), float(np.max(ys))
        if ymin == ymax:
            ymin -= 1; ymax += 1
        # grid lines
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            yy = y0 - (y0-y1)*frac
            draw.line([x0, yy, x1, yy], fill=grid, width=1)
        # y labels
        for frac in [0.0, 0.5, 1.0]:
            val = ymin + (ymax - ymin)*frac
            yy = y0 - (y0-y1)*frac
            draw.text((5, yy-8), fmt_usd(val), fill=fg, font=font)

        # line
        pts = []
        for x, v in zip(xs, ys):
            # normalize to [y1,y0]
            t = (v - ymin) / (ymax - ymin)
            yy = y0 - (y0-y1) * t
            pts.append((x, yy))
        if len(pts) >= 2:
            draw.line(pts, fill=line, width=3)

        # title
        title = "Equity Curve (Month)" if month_only else "Equity Curve (All)"
        draw.text((margin, 12), title, fill=fg, font=font_b)

        # popup
        top = tk.Toplevel(self)
        top.title(title)
        from PIL import ImageTk  # local import for this window
        cv = tk.Canvas(top, width=W, height=H, bg=bg, highlightthickness=0)
        cv.pack(fill=tk.BOTH, expand=True)
        self._eq_img = ImageTk.PhotoImage(img)
        cv.create_image(0,0, anchor="nw", image=self._eq_img)

    # ---------- Misc ----------
    def reset_filters(self):
        self.symbol_filter.set("")
        self.profits_only.set(False)
        self.firstweekday.set("SUN")
        self.refresh_calendar()

    def show_help(self):
        messagebox.showinfo("Shortcuts & Tips",
            "• Swipe left/right, or use ←/→, or Prev/Next to change months\n"
            "• Click a day to see per-stock P&L and trades\n"
            "• “n trades” shows number of orders that day\n"
            "• Symbol filter matches substrings (e.g., 'SOXL')\n"
            "• View → Dark/Light theme\n"
            "• Tools → Equity curve (All or Month)\n"
            "• File → Open Multiple to merge history files\n"
            "• Day Notes are saved in ~/.trade_calendar/notes.json\n"
            "• Settings are saved in ~/.trade_calendar/config.json"
        )

    def save_settings(self):
        self.cfg["symbol_filter"] = self.symbol_filter.get()
        self.cfg["profits_only"] = bool(self.profits_only.get())
        self.cfg["firstweekday"] = self.firstweekday.get()
        save_config(self.cfg)
        messagebox.showinfo("Saved", f"Settings saved to:\n{CONFIG_PATH}")

# ----------------------------- main -----------------------------
if __name__ == "__main__":
    App().mainloop()
