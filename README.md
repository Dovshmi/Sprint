# Webstock — Trade Calendar & Performance Dashboard

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Altair](https://img.shields.io/badge/Charts-Altair-0F9D58.svg)](https://altair-viz.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

A clean, fast Streamlit app for **Colmex Pro** traders to visualize filled orders as an interactive **calendar**, drill into **daily trades**, and track **weekly / monthly / yearly performance** with simple, beautiful charts.

> Current baseline: **v9** (`Webstock_v9.py`)

---

## ✨ Highlights

- **CSV Upload** for Colmex “Filled orders” export (`;` delimiter).
- **Clickable Calendar** (Sun→Sat) with color bubbles: 🟢 profit / 🔴 loss / ⚪ flat.
- **Hover tooltips** show **Winners** & **Losers** (symbols) for each day.
- **Trades Panel** for the selected day (side, quantity, price, gross, fees, net, etc.).
- **Performance Graphs**: **Weekly**, **Monthly**, **Yearly** (weekly aggregation).
  - Toggle **Line + Dots** or **Bars**.
  - Optional fees in tooltips.
  - Net (after fees) or Gross (before fees) basis.
- **Customizable Summary** (dropdown): YTD, totals, month P&L/fees, win rate, best/worst day, and more.
- **Dark‑theme friendly**, cached parsing for speed.

---

## 📸 Screenshots / GIFs

> Add these files under `assets/` and GitHub will render them automatically.

<p align="center">
  <img src="assets/calendar.png" alt="Calendar view with day bubbles" width="85%"><br/>
  <em>Clickable calendar — select a day to see its trades</em>
</p>

<p align="center">
  <img src="assets/weekly_chart.png" alt="Weekly performance chart" width="85%"><br/>
  <em>Weekly P&L — line + dots with winners/losers in tooltips</em>
</p>

<p align="center">
  <img src="assets/summary_popover.png" alt="Summary dropdown" width="70%"><br/>
  <em>Summary dropdown with customizable metrics</em>
</p>

<p align="center">
  <img src="assets/flow.gif" alt="Short demo" width="85%"><br/>
  <em>Quick flow demo (upload → calendar → charts)</em>
</p>

> Tip: To record a lightweight GIF on Windows, try **ShareX** or **ScreenToGif**. Keep it ~5–10 seconds for small file size.

---

## 🚀 Quickstart

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -U streamlit pandas altair

# 3) Run the app (use v9 as baseline)
streamlit run Webstock_v9.py
```

Now upload your **Colmex Pro – Filled orders** CSV and explore the calendar.

---

## 📂 CSV Format Notes (Colmex)

- Expected delimiter: `;`
- Required columns:
  - `Date/Time` (day‑first, e.g., `24.10.2025 16:31:42`)
  - `Gross P/L`, `Execution fee` *(or `Net P/L` — derived as `Gross - fee` if missing)*
- Useful columns (shown in the day grid when present): `Symbol`, `Side`, `Quantity`, `Price`, `Description`.

If parsing fails, the app will show a clear error. See **FAQ** below.

---

## 🧭 How it works

- **Calendar** is built from daily aggregation of P&L (net by default) and fees.
- **Tooltips** combine the symbols with positive and negative contribution for each day: **Winners / Losers**.
- **Weekly view** aggregates Sun→Sat; **Yearly view** shows weekly totals across the selected year.
- **Summary** is a dropdown with checkboxes — pick what you want to see (YTD, totals, month stats, best/worst, win rate, etc.).

---

## ⚙️ Configuration Hints

- **Fees in tooltips**: toggle with the “Show fees in tooltips” checkbox (Charts section).
- **Basis**: choose **Net (after fees)** or **Gross (before fees)** in both the chart area and in the summary dropdown.
- **First weekday**: calendar uses **Sunday**. To change it, edit `month_weeks_sunday_first()` in the source.
- **Theme**: Streamlit inherits your browser theme; the design is optimized for dark mode.

---

## 🧩 File Map

```
.
├── Webstock_v9.py          # Main app (baseline)
├── assets/                 # Place screenshots / gifs here
│   ├── calendar.png
│   ├── weekly_chart.png
│   ├── summary_popover.png
│   └── flow.gif
└── README.md               # This file
```

---

## 🐞 FAQ

**Q: “Failed to parse 'Date/Time'”**
- Ensure your CSV uses **day‑first** format (e.g., `29.10.2025 15:55:13`).
- The file should be exported as **Filled orders** from Colmex with `;` delimiter.

**Q: “Duplicate Streamlit key … day_YYYY-MM-DD”**
- This happens if multiple widgets share a key. The calendar uses unique keys by month and day. If you customized the code, check the `key=f"day_{month_key}_{d.isoformat()}"` lines.

**Q: My daily totals look off**  
- `Net P/L` is **derived** as `Gross P/L - Execution fee` when absent. Confirm your CSV headers and decimals.

**Q: Can I add a Daily chart?**  
- v9 intentionally focuses on Weekly/Monthly/Yearly. Daily chart can be added back on request.

---

## 🗺️ Roadmap

- [ ] Export to CSV/Excel (summary & per‑period aggregates)
- [ ] Compare month vs previous month
- [ ] Multi‑broker import templates (TradingView, IBKR, etc.)
- [ ] Per‑symbol performance heatmap
- [ ] Mobile layout refinements

---

## 🤝 Contributing

PRs are welcome! If you’re adding features or changing UX, please:
1. Open a short issue describing the change.
2. Keep the design minimal and dark‑theme friendly.
3. Add/update screenshots in `assets/` when relevant.

---

## 🧾 Changelog (excerpt)

- **v9**
  - Winners/Losers in day tooltips.
  - Weekly/Monthly/Yearly charts with **Line + Dots** or **Bars**.
  - Customizable **Summary** dropdown (YTD, totals, win rate, best/worst, etc.).
  - Restored **Trades on selected day** panel.
  - Removed “Daily” chart view.

---

## 📜 License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.
