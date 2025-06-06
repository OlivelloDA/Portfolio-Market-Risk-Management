# 📌 Portfolio Market Risk Management Equity/Equity Index investing portfolio

> Basic market risk management metrics are implemented to track the performance of an equity/equity index portfolio. This logic has been adapted for the Portfolio Positions Report that can be downloaded from the DeGiro brokerage platform at April 2025.
>
## 📖 About

This project is intended as a reference tool for retail investors who want to track their risk performance. It is based on a CSV file containing portfolio data exported from the Degiro broker in April 2025. You can run the analysis for your portfolio by downloading your portfolio data from Degiro and loading it into the Portfolio.csv file at https://github.com/OlivelloDA/Portfolio-Market-Risk-Management/src/portfolioAnalytics/data/

Example:
> Basic applications that calculate risk factor exposures, VaR, ES, Stress Tests.


## 🧠 Table of Contents
1. [Project Overview](#project-overview)
2. [Technical Architecture](#technical-architecture)
3. [Core Features](#core-features)
4. [Installation Guide](#installation-guide)
5. [Usage Examples](#usage-examples)
6. [Configuration Options](#configuration-options)
7. [Risk Methodologies](#risk-methodologies)
8. [Sample Outputs](#sample-outputs)
9. [Contributing Guidelines](#contributing-guidelines)
10. [License Information](#license-information)

---

## Project Overview

This enterprise-grade risk management system provides:
- Multi-asset portfolio risk analytics
- Regulatory-compliant risk reporting
- Stress testing and scenario analysis
- Real-time risk monitoring capabilities

Supported asset classes:
- Equities
- Fixed Income
- Derivatives
- FX
- Commodities

## Technical Architecture
```text
src/
├── main.py
├── portfolio_analysis/
│   ├── portfolio.py
│   ├── risk_metrics.py
│   ├── monte_carlo.py
│   └── stress_testing.py
├── data_management/
│   ├── data_loader.py
│   └── data_processor.py
└── visualization/
    ├── plotter.py
    └── report_generator.py
```

## Core Features

### Risk Measurement Suite
![Risk Methods Comparison](images/risk_methods.png)

- **Value-at-Risk (VaR)**
  - Parametric (Delta-Normal)
  - Historical Simulation
  - Monte Carlo (GBM, Jump Diffusion)
  
- **Expected Shortfall (CVaR)**
- **Incremental VaR**
- **Component VaR**
- **Stress VaR**

### Portfolio Analysis Tools
- Concentration Risk
- Liquidity Risk
- Factor Exposure
- Scenario Analysis

## Installation Guide

### Prerequisites
- Python 3.8+
- pip 20.0+

### Setup

## Clone repository
git clone https://github.com/OlivelloDA/Portfolio-Market-Risk-Management.git
cd Portfolio-Market-Risk-Management

## Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

## Install dependencies
pip install -r requirements.txt

## Verify installation
python -c "import risk_metrics; print('Installation successful')"

# Usage Examples

## Basic Risk Report
```text
from portfolio_analysis import Portfolio
from risk_metrics import calculate_var

portfolio = Portfolio.from_config('config/portfolio.json')
var_results = calculate_var(portfolio, method='monte_carlo')
var_results.save_report('output/risk_report.html')
```
## Stress Testing
```text
from stress_testing import run_stress_scenarios

scenarios = ['2008_crisis', '2020_covid', 'rate_shock+200bp']
results = run_stress_scenarios(portfolio, scenarios)
results.plot_impact()
```
# Configuration Options

## Portfolio Specification (JSON)
```text
{
  "name": "Global Balanced Portfolio",
  "base_currency": "USD",
  "positions": [
    {
      "asset_id": "AAPL.US",  
      "quantity": 1000,
      "asset_type": "equity"
    },
    {
      "asset_id": "US10Y.GBL",
      "notional": 500000,
      "asset_type": "bond"
    }
  ]
}
```
## Risk Parameters
```text
var:
  confidence_level: 0.99
  horizon_days: 10
  methods: [parametric, historical, monte_carlo]

monte_carlo:
  simulations: 10000
  model: "Heston"
  random_seed: 42
```
# Risk Methodologies
## Parametric VaR
```text
VaR = -P * (μ + z * σ)
Where:
P = Portfolio value
μ = Mean return
z = Z-score for confidence level
σ = Portfolio volatility
```
## Monte Carlo Simulation

1. Calibrate stochastic processes
2. Generate correlated random paths
3. Revalue portfolio under each scenario
4. Calculate percentile losses

## Stress Testing Framework

1. Historical scenarios
2. Hypothetical shocks
3. Reverse stress tests
4. Plausibility checks

# MIT License
Copyright (c) [2025] [OlivelloDA]




