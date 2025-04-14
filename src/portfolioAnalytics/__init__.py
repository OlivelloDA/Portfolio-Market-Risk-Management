# portfolio_analyzer/__init__.py

"""
Portfolio Analyzer Package

A comprehensive suite of tools for analyzing investment portfolios,
including factor analysis, risk metrics (VaR, ES), and stress testing.
"""

# Import main classes to make them directly available from the package
from .stress_test import StressTest
from .factor_analysis import FactorAnalysis
from .var_model import VaRModel
from .expected_shortfall import ExpectedShortfall

# Package metadata
__version__ = '0.1.0'
__author__ = 'Portfolio Analysis Team'

# Define what's available when using 'from portfolio_analyzer import *'
__all__ = [
    'StressTest',
    'FactorAnalysis',
    'VaRModel',
    'ExpectedShortfall'
]


# Optional: Add any package-level helper functions
def load_portfolio(filepath="Portfolio.csv"):
    """
    Helper function to load a portfolio from a CSV file

    Parameters:
    -----------
    filepath : str
        Path to the portfolio CSV file

    Returns:
    --------
    pandas.DataFrame
        Loaded portfolio data
    """
    import pandas as pd
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Portfolio file not found at {filepath}")
    except Exception as e:
        raise Exception(f"Error loading portfolio: {str(e)}")


def run_full_analysis(portfolio_df=None, filepath=None, confidence=0.99,
                      time_horizons=[1, 5, 10], stress_scenarios=[-0.05, -0.10, -0.20, -0.30]):
    """
    Run a complete portfolio analysis including factor analysis, VaR, ES, and stress tests

    Parameters:
    -----------
    portfolio_df : pandas.DataFrame or None
        Portfolio data. If None, will attempt to load from filepath
    filepath : str or None
        Path to portfolio CSV file if portfolio_df is None
    confidence : float
        Confidence level for VaR and ES calculations (default: 0.99)
    time_horizons : list
        Time horizons in days for VaR and ES calculations
    stress_scenarios : list
        Market decline scenarios for stress testing (as decimal percentages)

    Returns:
    --------
    dict
        Dictionary containing all analysis results
    """
    if portfolio_df is None:
        if filepath is None:
            filepath = "Portfolio.csv"
        portfolio_df = load_portfolio(filepath)

    # Run all analyses
    factor_analyzer = FactorAnalysis(portfolio_df)
    var_model = VaRModel(portfolio_df)
    es_model = ExpectedShortfall(portfolio_df)
    stress_tester = StressTest(portfolio_df)

    # Collect results
    results = {
        "factor_analysis": factor_analyzer.analyze_factors(),
        "var_results": {
            horizon: var_model.calculate_var(confidence=confidence, time_horizon=horizon)
            for horizon in time_horizons
        },
        "es_results": {
            horizon: es_model.calculate_expected_shortfall(confidence=confidence, time_horizon=horizon)
            for horizon in time_horizons
        },
        "stress_test": stress_tester.run_market_stress_test(sp500_scenarios=stress_scenarios)
    }

    return results