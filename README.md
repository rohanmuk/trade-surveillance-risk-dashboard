# Trade Surveillance Risk Dashboard

## Project Overview

This project is a Python-based trade surveillance dashboard designed to identify unusual trading activity and operational risk indicators across a simulated trading dataset.

The dashboard analyzes trade-level data to flag patterns such as large notional trades, unusual trade volumes, concentrated counterparty exposure, late trade amendments, and potential trade monitoring exceptions.

This project was built as a personal portfolio project to demonstrate skills in financial markets analytics, risk monitoring, data visualization, and Python-based dashboard development.

## Business Problem

Trading and capital markets teams process large volumes of transactions across products, counterparties, traders, and desks. Risk, compliance, and operations teams need tools to monitor trading activity and identify transactions that may require review.

This dashboard simulates that process by applying rule-based surveillance checks to trade data and presenting the results in an interactive format.

## Key Features

- Interactive trade surveillance dashboard
- Large notional trade flagging
- Counterparty concentration analysis
- Trader-level and desk-level exposure monitoring
- Trade volume trend analysis
- Late amendment and cancellation indicators
- Exception summary by product, desk, and counterparty
- Visual risk indicators using charts and tables

## Tools Used

- Python
- pandas
- NumPy
- Streamlit
- Plotly
- SQL / SQLite
- GitHub

## Dataset

This project uses a simulated trade dataset. No real client, counterparty, employee, or employer data is used.

Example fields include:

- Trade ID
- Trade Date
- Trader
- Desk
- Product Type
- Counterparty
- Buy/Sell Indicator
- Quantity
- Price
- Notional Value
- Trade Status
- Amendment Flag
- Cancellation Flag
- Settlement Date

## Surveillance Logic

The dashboard applies sample rule-based checks, including:

1. **Large Notional Trade Flag**
   - Flags trades above a defined notional threshold.

2. **Counterparty Concentration Flag**
   - Identifies counterparties with unusually high exposure.

3. **High Trade Volume Flag**
   - Identifies traders or desks with unusually high trading activity.

4. **Amendment / Cancellation Flag**
   - Flags trades that were amended or cancelled after booking.

5. **Settlement Risk Indicator**
   - Highlights trades with potential settlement timing concerns.

## Dashboard Pages

The dashboard includes:

- Executive Summary
- Trade Exceptions
- Counterparty Exposure
- Trader / Desk Activity
- Product-Level Risk Analysis
- Raw Trade Data

## Example Use Case

A risk analyst can use this dashboard to quickly identify which trades, desks, counterparties, or products may require further review.

For example, if a specific counterparty has a high percentage of total exposure, or if a trader has an unusually high number of late amendments, the dashboard highlights these items for investigation.

## Project Structure

```text
trade-surveillance-risk-dashboard/
│
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── simulated_trades.csv
│
├── notebooks/
│   └── trade_surveillance_eda.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── surveillance_rules.py
│   └── visualizations.py
│
└── images/
    └── dashboard_preview.png
