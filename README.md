# Trade Surveillance Risk Dashboard

A Streamlit dashboard that reads a trade blotter and surfaces the activity a risk,
compliance or operations analyst would want to look at first: oversized tickets,
concentrated counterparty exposure, hyperactive traders and desks, lifecycle events booked
long after the trade, and settlement dates that fall outside market convention.

> **All data in this repository is simulated.** Trader and counterparty names are generated
> with [Faker](https://faker.readthedocs.io/). There is no real client, counterparty or
> employee data anywhere in the project.

![Dashboard preview](images/dashboard_preview.png)

<!-- The preview image is captured by hand -- see images/README.md. -->

---

## What it does

The blotter is 3,600 simulated trades over six months (Jan–Jun 2026) across five desks,
five product types, fifteen traders and twenty-five counterparties — about **$25.2bn** of
gross notional. Five surveillance rules run over it:

| # | Rule | Flags | Key thresholds |
|---|------|-------|----------------|
| 1 | **Large notional** | Trades above a size cut-off, set either as a percentile of the book or as a fixed dollar amount | `percentile` (default p99) or `threshold` |
| 2 | **Counterparty concentration** | Counterparties above a share-of-notional threshold, or inside the top N by notional | `exposure_pct_threshold`, `top_n` |
| 3 | **High trade volume** | Traders and desks whose activity is an outlier within their peer group, by z-score or Tukey IQR fence | `z_threshold`, `iqr_multiplier`, `metric`, `method` |
| 4 | **Late amendment / cancellation** | Lifecycle events booked more than N business days after the trade date | `business_days_threshold` |
| 5 | **Settlement risk** | Trades settling outside the expected window for their product — too fast, too slow, or dated before the trade | per-product windows, or flat `min_days`/`max_days` |

Every threshold is a function argument, not a constant buried in the rule body, which is
what lets the sidebar expose all of them as live controls.

### Trade-level vs entity-level exceptions

The dashboard reports two exception rates rather than one, and the distinction is the most
important design decision in the project.

- **Trade-level** rules (1, 4, 5) describe an individual ticket. They flag **3.6%** of the
  book — roughly one ticket a day, a queue an ops team can actually work through.
- **Entity-level** rules (2, 3) describe a *counterparty, trader or desk*, so they
  necessarily tag every trade that entity touched. They flag **57.8%** of the book, almost
  all of it from one busy desk and the top five counterparties.

Adding those together and calling the result a 59% "exception rate" would be actively
misleading — it is the classic way a surveillance system generates alert fatigue. The
Executive Summary reports the two separately and treats entity-level findings as flagged
*entities*, not flagged trades.

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

The app opens on <http://localhost:8501>. The committed dataset
(`data/simulated_trades.csv`) is read on startup — nothing else needs to be generated
first.

### Run the tests

```bash
python -m pytest
```

82 tests covering loading, validation, cleaning, all five rules, the orchestrator, the
chart builders and the SQL views.

### Explore the notebook

```bash
jupyter notebook notebooks/trade_surveillance_eda.ipynb
```

The committed copy is pre-executed, so all outputs are visible without running anything.
Its charts are interactive Plotly figures that load plotly.js from a CDN — they render in
Jupyter or nbviewer, but GitHub's static notebook preview will show them as blank.

---

## How to regenerate the dataset

```bash
python scripts/generate_trades.py
```

This rewrites `data/simulated_trades.csv`. The seed is fixed
(`RANDOM_SEED` in `src/config.py`), so re-running reproduces the committed file byte for
byte. To vary it:

```bash
python scripts/generate_trades.py --rows 10000 --seed 42 --out data/bigger_blotter.csv
```

The generator deliberately plants the patterns the rules are meant to catch — a skewed
counterparty distribution, uneven trader and desk activity, a 1.5% tail of trades sized
10–50x normal, ~15% of lifecycle events booked late, and a handful of impossible
settlement dates. Without them the dashboard would be a page of zeroes.

---

## Project structure

```
trade-surveillance-risk-dashboard/
├── README.md
├── app.py                      Streamlit app: layout, sidebar, filtering only
├── requirements.txt
├── .gitignore
├── .streamlit/config.toml      Light theme pinned to match the chart palette
│
├── data/
│   └── simulated_trades.csv    Committed sample blotter (3,600 rows)
│
├── scripts/
│   └── generate_trades.py      Seeded synthetic data generator
│
├── notebooks/
│   └── trade_surveillance_eda.ipynb
│
├── src/
│   ├── config.py               Paths, domain constants, default thresholds, palette
│   ├── data_cleaning.py        load_trades / clean_trades + CleaningReport
│   ├── surveillance_rules.py   The five rules + run_all_rules orchestrator
│   ├── visualizations.py       Plotly figure builders (no Streamlit import)
│   └── sqlite_views.py         SQL-backed aggregates over the flagged blotter
│
├── tests/
│   ├── conftest.py             10-row hand-built blotter with one of each pathology
│   ├── test_data_cleaning.py
│   ├── test_surveillance_rules.py
│   └── test_visualizations_and_sql.py
│
└── images/
    └── dashboard_preview.png   (capture by hand — see images/README.md)
```

`app.py` owns layout and nothing else. Every number it displays comes from `src/`, which
means the notebook and the test suite reproduce the dashboard's figures exactly.

---

## The pages

| Page | What's on it |
|------|--------------|
| **Executive Summary** | KPI row, trade-volume trend, exceptions by desk, counterparty exposure, and a SQL-backed desk league table |
| **Trade Exceptions** | Every flagged trade, filterable by rule, trader and any/all matching, with CSV export |
| **Counterparty Exposure** | Concentration chart with cumulative-share overlay, full exposure table, HHI, and the same view expressed in SQL |
| **Trader / Desk Activity** | Peer-group outlier detection with the z-score ceiling guard, activity chart, and per-desk trend |
| **Product-Level Risk Analysis** | Exceptions and notional by product, settlement-timing distribution, per-product notional thresholds, lifecycle-event summary |
| **Raw Trade Data** | Full searchable blotter with a column picker and CSV export |

Sidebar filters (date range, desk, product) are applied **before** the rules run. That is
deliberate: a percentile threshold, a concentration share and a peer-group z-score are all
*relative* measures, so computing them on the full book while displaying a filtered slice
would put mutually inconsistent numbers on the same screen.

---

## Notes on the analytics

A few things that came out of building this and are worth knowing before you change a
threshold.

**The z-score has a hard ceiling on small peer groups.** With a sample standard deviation,
the largest attainable z-score across *n* groups is `sqrt(n - 1)`. For five desks that is
exactly **2.00** — so the 2.0 threshold that works fine for fifteen traders can never fire
on desks, no matter how lopsided the book is. The app defaults desks to 1.2, displays the
ceiling next to the flag counts, and warns if you set the threshold above it. The IQR
method is the more robust alternative for small groups.

**Settlement windows have to be product-aware.** FX Forwards in this book settle one to
three months out by construction. A flat T+1..T+3 test would flag all 710 of them and tell
an analyst nothing; the per-product windows in `src/config.py` cut that to about 40 genuine
outliers, including a few tickets dated to settle *before* they were traded.

**A fixed dollar threshold cannot serve the whole book.** The median ticket is ~$2.4m and
the largest is ~$750m, a spread of more than two orders of magnitude across products. The
percentile mode adapts; `flag_large_notional(..., by_product=True)` goes further and judges
each product against its own distribution.

**Concentration needs more than one lens.** The HHI of this book is ~660, which on the
merger-guideline reading is *unconcentrated* — yet the largest single counterparty faces
16% of gross notional, which would sit at or above a typical single-name limit. The rule
therefore flags on share and on rank, and the dashboard shows the HHI alongside rather than
in place of them.

**Why SQLite.** The dashboard is a pandas application, so the SQL is not there to do work
pandas cannot. It is there because these aggregates are the ones that would live as saved
views in a surveillance warehouse, and an analyst reading the desk league table should be
able to see the query that produced it — every SQL-backed table on the app has a "Show the
SQL" expander next to it.

---

## Data dictionary

| Column | Type | Notes |
|---|---|---|
| `trade_id` | str | Unique, `T00001`-style, issued in booking order |
| `trade_date` | date | Business days only, Jan–Jun 2026 |
| `trader` | str | One of 15 synthetic names; each belongs to one desk |
| `desk` | str | Rates, FX, Credit, Equities, Commodities |
| `product_type` | str | Bond, Swap, FX Forward, Equity, Option |
| `counterparty` | str | One of 25 synthetic names, share ∝ `1 / rank ** 0.6` |
| `buy_sell` | str | Buy / Sell |
| `quantity` | int | Product-appropriate units |
| `price` | float | Product-appropriate unit price |
| `notional_value` | float | `quantity × price`, with an injected 1.5% outlier tail |
| `trade_status` | str | Booked / Amended / Cancelled |
| `amendment_flag` | bool | True when the trade was modified after booking. **A cancellation is a post-booking modification, so cancelled trades carry this flag too** — `trade_status` is what distinguishes the two event types |
| `cancellation_flag` | bool | True when `trade_status == "Cancelled"` |
| `booking_datetime` | datetime | When the ticket was booked |
| `amendment_datetime` | datetime? | Timestamp of the amendment or cancellation; null for Booked trades |
| `settlement_date` | date | Per-product convention, with a small number of injected anomalies |

Cleaning derives three more: `trade_month`, `days_to_settlement` and `days_to_amendment`.
Both day counts are **business days** (`numpy.busday_count`, Mon–Fri, no holiday calendar)
and are signed, so a settlement date before the trade date reads as negative — which is
exactly what the settlement rule looks for.

---

## Requirements

Python 3.9+ (developed and tested on 3.9; runs unchanged on 3.11/3.12). Dependencies are
pinned in `requirements.txt`: pandas, NumPy, Streamlit, Plotly, Faker, notebook and pytest.
SQLite comes from the standard library.

## Licence

MIT.
