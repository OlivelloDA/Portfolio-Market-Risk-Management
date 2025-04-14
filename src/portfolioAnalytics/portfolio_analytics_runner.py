# portfolio_analysis.py
# Main script to run all portfolio analyses

import pandas as pd
import numpy as np
from pathlib import Path

from portfolio_loader import PortfolioLoader
from portfolio_factor_analysis import FactorAnalyzer
from portfolio_risk_management import RiskManager
from portfolio_stress_test import StressTest


def main():
    """
    Main function to run the portfolio analysis pipeline:
    1. Load and preprocess the portfolio data
    2. Perform factor analysis to understand exposure to different risk factors
    3. Calculate risk metrics (VaR, ES) at 99% confidence level
    4. Conduct stress tests for S&P 500 drawdown scenarios
    """
    # Load portfolio data
    print("Loading portfolio data...")
    loader = PortfolioLoader('data/Portfolio.csv')
    portfolio = loader.load_portfolio()

    # Display portfolio summary
    print("\nPortfolio Summary:")
    print(f"Total Portfolio Value: €{portfolio['total_value']:.2f}")
    print(f"Number of Assets: {portfolio['num_assets']}")
    print(f"Currency Exposure: {portfolio['currency_exposure']}")

    # Perform factor analysis
    print("\nPerforming Factor Analysis...")
    factor_analyzer = FactorAnalyzer(portfolio['data'])
    factor_exposures = factor_analyzer.analyze_factors()
    factor_analyzer.display_factor_exposures()

    # Risk management analysis
    print("\nCalculating Risk Metrics...")
    risk_manager = RiskManager(portfolio['data'])

    # Calculate VaR and ES at 99% confidence level for different horizons
    horizons = [1, 5, 10, 20, 60]
    for horizon in horizons:
        var, es = risk_manager.calculate_risk_metrics(confidence_level=0.99, time_horizon=horizon)
        print(f"Time Horizon: {horizon} days")
        print(f"Value at Risk (99%): €{var:.2f} ({var / portfolio['total_value'] * 100:.2f}% of portfolio)")
        print(f"Expected Shortfall (99%): €{es:.2f} ({es / portfolio['total_value'] * 100:.2f}% of portfolio)")

    # Perform stress tests
    print("\nRunning Stress Tests...")
    stress_tester = StressTest(portfolio['data'])
    drawdowns = [-0.05, -0.10, -0.20, -0.30]

    for drawdown in drawdowns:
        results_df, category_impacts_df = stress_tester.run_market_stress_test([drawdown])
        # Get the impact value from the results DataFrame
        scenario_key = f"SP500_{int(abs(drawdown * 100))}pct_down"
        impact = results_df.loc[scenario_key, "Portfolio_Impact_EUR"]

        print(f"Portfolio Impact: €{impact:.2f} ({impact / portfolio['total_value'] * 100:.2f}% of portfolio)")


if __name__ == "__main__":
    main()