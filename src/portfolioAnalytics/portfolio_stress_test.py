# stress_test.py

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class StressTest:
    """
    Class for performing stress tests on a portfolio under various market scenarios.

    The stress test simulates portfolio performance under specific adverse market
    conditions, particularly focusing on equity market downturns of varying severity.
    """

    def __init__(self, portfolio_df, asset_returns_corr=None, sp500_correlations=None):
        """
        Initialize the StressTest object with portfolio data and correlation assumptions.

        Parameters:
        -----------
        portfolio_df : pandas.DataFrame
            DataFrame containing portfolio positions with columns:
            - Prodotto (Product name)
            - Codice (Ticker)
            - Quantità (Quantity)
            - Valore in EUR (Value in EUR)
        asset_returns_corr : pandas.DataFrame or None
            Correlation matrix of asset returns. If None, estimated correlations will be used.
        sp500_correlations : pandas.Series or None
            Correlations between individual assets and S&P500. If None, estimated values will be used.
        """
        self.portfolio_df = portfolio_df
        self.total_value = portfolio_df["Valore in EUR"].astype(float).sum()

        # Clean the portfolio data
        self.portfolio_df["Weight"] = self.portfolio_df["Valore in EUR"] / self.portfolio_df["Valore in EUR"].sum()

        # Set up correlation matrices (if not provided, we'll use estimates)
        self.asset_returns_corr = self._estimate_asset_correlations() if asset_returns_corr is None else asset_returns_corr
        self.sp500_correlations = self._estimate_sp500_correlations() if sp500_correlations is None else sp500_correlations

        # Asset categories for better stress test modeling
        self.categorize_assets()

    def _estimate_asset_correlations(self):
        """
        Estimate correlation matrix between assets based on typical market relationships.

        Returns:
        --------
        pandas.DataFrame
            Estimated correlation matrix for the assets in the portfolio

        Note:
        -----
        In a real-world implementation, this would use historical return data.
        This implementation uses reasonable approximations based on asset classes.
        """
        assets = self.portfolio_df["Codice"].tolist()
        n_assets = len(assets)

        # Create a base correlation matrix (identity matrix)
        corr_matrix = np.eye(n_assets)

        # Set realistic correlation values based on asset types
        # This is a simplified model - in production, use historical data
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                # Default moderate correlation
                corr_value = 0.3

                # Higher correlation for assets in similar sectors/classes
                ticker_i = str(self.portfolio_df.iloc[i]["Codice"])
                ticker_j = str(self.portfolio_df.iloc[j]["Codice"])

                # Check if both are ETFs
                if ("ISHARES" in str(self.portfolio_df.iloc[i]["Prodotto"]) and
                        "ISHARES" in str(self.portfolio_df.iloc[j]["Prodotto"])):
                    corr_value = 0.75

                # Check if both are tech stocks
                elif any(tech in str(self.portfolio_df.iloc[i]["Prodotto"]).upper() for tech in
                         ["ALPHABET", "INTEL", "ZOOM"]) and \
                        any(tech in str(self.portfolio_df.iloc[j]["Prodotto"]).upper() for tech in
                            ["ALPHABET", "INTEL", "ZOOM"]):
                    corr_value = 0.65

                # European stocks correlation
                elif any(euro in str(self.portfolio_df.iloc[i]["Prodotto"]).upper() for euro in
                         ["ASML", "AIRBUS", "DIASORIN"]) and \
                        any(euro in str(self.portfolio_df.iloc[j]["Prodotto"]).upper() for euro in
                            ["ASML", "AIRBUS", "DIASORIN"]):
                    corr_value = 0.6

                corr_matrix[i, j] = corr_value
                corr_matrix[j, i] = corr_value  # Symmetric

        return pd.DataFrame(corr_matrix, index=assets, columns=assets)

    def _estimate_sp500_correlations(self):
        """
        Estimate correlations between portfolio assets and S&P500.

        Returns:
        --------
        pandas.Series
            Estimated correlations with S&P500 for each asset

        Note:
        -----
        In a real-world implementation, this would use historical return data.
        This implementation uses reasonable approximations based on asset type.
        """
        # Create a series with reasonable correlation estimates
        correlations = {}

        for idx, row in self.portfolio_df.iterrows():
            ticker = row["Codice"]
            product = str(row["Prodotto"]).upper()

            # US Large Cap stocks have high correlation with S&P500
            if any(us_stock in product for us_stock in ["ALPHABET", "INTEL", "DISNEY", "KRAFT"]):
                correlations[ticker] = 0.80

            # ETFs correlation depends on their underlying
            elif "ISHARES" in product or "VANGUARD" in product:
                if "TREASURY" in product:  # Treasury bonds typically have negative correlation
                    correlations[ticker] = -0.20
                elif "WORLD" in product:  # Global ETFs have high correlation
                    correlations[ticker] = 0.85
                else:  # Other ETFs moderate correlation
                    correlations[ticker] = 0.60

            # European stocks have moderate correlation with S&P500
            elif any(euro in product for euro in ["ASML", "AIRBUS", "DIASORIN"]):
                correlations[ticker] = 0.65

            # Small caps and biotech have lower correlation
            elif any(small in product for small in ["ADAPTIMMUNE", "GALAPAGOS", "AETERNA"]):
                correlations[ticker] = 0.40

            # Cash has no correlation
            elif "CASH" in product:
                correlations[ticker] = 0.0

            # Default moderate correlation
            else:
                correlations[ticker] = 0.50

        return pd.Series(correlations)

    def categorize_assets(self):
        """
        Categorize assets in the portfolio for better stress modeling.

        This method adds category and beta information to each asset, which allows
        for more accurate stress testing by modeling different asset classes differently.
        """
        # Create category and beta columns
        self.portfolio_df["Category"] = "Other"
        self.portfolio_df["Beta"] = 1.0

        # Categorize based on product name and adjust beta
        for idx, row in self.portfolio_df.iterrows():
            product = str(row["Prodotto"]).upper()

            # Cash and cash equivalents
            if "CASH" in product:
                self.portfolio_df.at[idx, "Category"] = "Cash"
                self.portfolio_df.at[idx, "Beta"] = 0.0

            # US Treasury bonds
            elif "TREASURY" in product:
                self.portfolio_df.at[idx, "Category"] = "Treasury"
                self.portfolio_df.at[idx, "Beta"] = -0.2

            # US Large Cap Tech
            elif any(tech in product for tech in ["ALPHABET", "INTEL", "ZOOM"]):
                self.portfolio_df.at[idx, "Category"] = "US Tech"
                self.portfolio_df.at[idx, "Beta"] = 1.2

            # US Consumer
            elif any(consumer in product for consumer in ["DISNEY", "KRAFT"]):
                self.portfolio_df.at[idx, "Category"] = "US Consumer"
                self.portfolio_df.at[idx, "Beta"] = 0.9

            # European Large Cap
            elif any(euro in product for euro in ["ASML", "AIRBUS", "DIASORIN", "MONCLER"]):
                self.portfolio_df.at[idx, "Category"] = "European Equity"
                self.portfolio_df.at[idx, "Beta"] = 0.8

            # Healthcare/Biotech
            elif any(health in product for health in ["HEALTH", "ADAPTIMMUNE", "GALAPAGOS"]):
                self.portfolio_df.at[idx, "Category"] = "Healthcare"
                self.portfolio_df.at[idx, "Beta"] = 0.7

            # World ETFs
            elif "WORLD" in product:
                self.portfolio_df.at[idx, "Category"] = "World ETF"
                self.portfolio_df.at[idx, "Beta"] = 0.9

            # Specialty ETFs
            elif any(specialty in product for specialty in ["URANIUM", "WATER"]):
                self.portfolio_df.at[idx, "Category"] = "Specialty ETF"
                self.portfolio_df.at[idx, "Beta"] = 0.85

            # Emerging Markets
            elif "EM" in product or "SAUDI" in product:
                self.portfolio_df.at[idx, "Category"] = "Emerging Markets"
                self.portfolio_df.at[idx, "Beta"] = 1.1

    def run_market_stress_test(self, sp500_scenarios=[-0.05, -0.10, -0.20, -0.30]):
        """
        Run stress tests for given S&P500 drawdown scenarios.

        Parameters:
        -----------
        sp500_scenarios : list
            List of S&P500 return scenarios to test (e.g., [-0.05, -0.10, -0.20, -0.30])

        Returns:
        --------
        pandas.DataFrame
            Results of the stress tests showing:
            - Portfolio value impact for each scenario
            - Percentage loss for each scenario
            - Value at risk
            - Asset class contributions to losses

        Note:
        -----
        The method uses beta-adjusted correlations to model how each asset
        would respond to market stress conditions based on its characteristics.
        """

        # Convert single float to list if needed
        if isinstance(sp500_scenarios, (float, int)):
            sp500_scenarios = [sp500_scenarios]

        results = {}
        category_impacts = {}

        # For each scenario, calculate the expected impact
        for scenario in sp500_scenarios:
            asset_impacts = []
            scenario_cat_impacts = {}

            # Calculate impact on each asset
            for idx, asset in self.portfolio_df.iterrows():
                ticker = asset["Codice"]
                value = asset["Valore in EUR"]
                beta = asset["Beta"]
                category = asset["Category"]

                # Get correlation with S&P500 (or use beta as an approximation)
                if ticker in self.sp500_correlations:
                    corr = self.sp500_correlations[ticker]
                else:
                    corr = beta * 0.8  # Approximation using beta

                # Calculate expected return given the scenario
                # Formula: Asset Return = Beta * Correlation * Market Return + Idiosyncratic Return
                # We'll simplify and not model idiosyncratic return in this implementation
                expected_return = beta * corr * scenario

                # Calculate impact
                impact = value * expected_return
                asset_impacts.append(impact)

                # Aggregate by category
                if category not in scenario_cat_impacts:
                    scenario_cat_impacts[category] = 0
                scenario_cat_impacts[category] += impact

            # Calculate overall portfolio impact
            total_impact = sum(asset_impacts)
            pct_impact = total_impact / self.total_value

            results[f"SP500_{int(abs(scenario * 100))}pct_down"] = {
                "SP500_Return": scenario,
                "Portfolio_Impact_EUR": total_impact,
                "Portfolio_Return_Pct": pct_impact,
                "New_Portfolio_Value": self.total_value + total_impact
            }

            category_impacts[f"SP500_{int(abs(scenario * 100))}pct_down"] = scenario_cat_impacts

        # Convert results to DataFrame
        results_df = pd.DataFrame(results).T
        category_impacts_df = pd.DataFrame(category_impacts).T

        return results_df, category_impacts_df

    def plot_stress_test_results(self, results_df, category_impacts_df, save_path=None):
        """
        Create visualizations of stress test results.

        Parameters:
        -----------
        results_df : pandas.DataFrame
            Overall stress test results from run_market_stress_test
        category_impacts_df : pandas.DataFrame
            Category-level impacts from run_market_stress_test
        save_path : str or None
            If provided, save the plots to this path

        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the stress test visualizations
        """
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Portfolio value under stress scenarios
        scenarios = results_df.index
        baseline = [self.total_value] * len(scenarios)
        new_values = results_df["New_Portfolio_Value"].values

        ax1 = axes[0]
        ax1.bar(scenarios, baseline, label="Original Value", color="green", alpha=0.6)
        ax1.bar(scenarios, new_values, label="Stressed Value", color="red", alpha=0.6)

        # Add value labels
        for i, scenario in enumerate(scenarios):
            original = self.total_value
            new_val = new_values[i]
            loss_pct = results_df.loc[scenario, "Portfolio_Return_Pct"] * 100

            ax1.text(i, original / 2, f"{original:.2f} EUR", ha="center", va="center", color="black")
            ax1.text(i, new_val / 2, f"{new_val:.2f} EUR\n({loss_pct:.2f}%)", ha="center", va="center", color="black")

        ax1.set_title("Portfolio Value Under Stress Scenarios", fontsize=14)
        ax1.set_ylabel("Portfolio Value (EUR)")
        ax1.legend()

        # Category contributions to losses
        category_impacts_df_pct = category_impacts_df.div(category_impacts_df.sum(axis=1), axis=0) * 100

        ax2 = axes[1]
        category_impacts_df_pct.plot(kind="bar", stacked=True, ax=ax2, colormap="viridis")
        ax2.set_title("Category Contributions to Losses by Scenario", fontsize=14)
        ax2.set_ylabel("Contribution to Loss (%)")
        ax2.set_xlabel("Stress Scenario")
        ax2.legend(title="Asset Category", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Adjust layout
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def run_correlation_break_stress_test(self, correlation_scenarios=[0.5, 0.7, 0.9]):
        """
        Run stress tests that simulate correlation breakdowns during market stress.

        Parameters:
        -----------
        correlation_scenarios : list
            List of correlation multipliers to test (e.g., [0.5, 0.7, 0.9])
            where 1.0 means correlations remain unchanged, and values > 1.0
            mean correlations increase during stress (typical in market crises)

        Returns:
        --------
        pandas.DataFrame
            Results showing portfolio impact under different correlation scenarios

        Note:
        -----
        During market crises, correlations often increase ("correlation breakdown"),
        which can reduce diversification benefits. This test simulates that effect.
        """
        # Base scenario: 20% market decline
        sp500_scenario = -0.20
        results = {}

        # First run the normal scenario as a baseline
        normal_results, _ = self.run_market_stress_test([sp500_scenario])
        normal_impact = normal_results.iloc[0]["Portfolio_Impact_EUR"]
        normal_return = normal_results.iloc[0]["Portfolio_Return_Pct"]

        results["Normal"] = {
            "SP500_Return": sp500_scenario,
            "Correlation_Modifier": 1.0,
            "Portfolio_Impact_EUR": normal_impact,
            "Portfolio_Return_Pct": normal_return
        }

        # Run scenarios with increased correlations
        for corr_mod in correlation_scenarios:
            asset_impacts = []

            # Calculate impact on each asset with modified correlations
            for idx, asset in self.portfolio_df.iterrows():
                ticker = asset["Codice"]
                value = asset["Valore in EUR"]
                beta = asset["Beta"]

                # Get correlation with S&P500 and modify it
                if ticker in self.sp500_correlations:
                    base_corr = self.sp500_correlations[ticker]
                else:
                    base_corr = beta * 0.8  # Approximation

                # Modify correlation (clamp to [-1, 1])
                modified_corr = min(max(base_corr * corr_mod, -1.0), 1.0)

                # Calculate expected return
                expected_return = beta * modified_corr * sp500_scenario

                # Calculate impact
                impact = value * expected_return
                asset_impacts.append(impact)

            # Calculate overall portfolio impact
            total_impact = sum(asset_impacts)
            pct_impact = total_impact / self.total_value

            results[f"Corr_modifier_{corr_mod}"] = {
                "SP500_Return": sp500_scenario,
                "Correlation_Modifier": corr_mod,
                "Portfolio_Impact_EUR": total_impact,
                "Portfolio_Return_Pct": pct_impact,
                "New_Portfolio_Value": self.total_value + total_impact
            }

        # Convert results to DataFrame
        results_df = pd.DataFrame(results).T

        return results_df

    def run_liquidity_stress_test(self, liquidity_haircuts=None):
        """
        Run a stress test that accounts for liquidity issues during market stress.

        Parameters:
        -----------
        liquidity_haircuts : dict or None
            Dictionary mapping asset categories to liquidity haircuts (additional losses)
            If None, default values will be used

        Returns:
        --------
        pandas.DataFrame
            Results showing portfolio impact under liquidity stress

        Note:
        -----
        During market stress, less liquid assets often face steeper price declines
        due to wider bid-ask spreads and market impact costs. This test simulates
        those additional losses beyond what correlation-based models predict.
        """
        if liquidity_haircuts is None:
            # Default liquidity haircuts by category (additional loss percentages)
            liquidity_haircuts = {
                "Cash": 0.00,  # Cash is fully liquid
                "Treasury": 0.01,  # Treasuries very liquid but some impact
                "US Tech": 0.03,  # Large cap tech fairly liquid
                "US Consumer": 0.04,  # Large cap consumer stocks
                "European Equity": 0.05,  # European stocks slightly less liquid
                "Healthcare": 0.08,  # Healthcare/biotech less liquid
                "World ETF": 0.02,  # World ETFs quite liquid
                "Specialty ETF": 0.06,  # Specialty ETFs less liquid
                "Emerging Markets": 0.10,  # EM less liquid in stress
                "Other": 0.07  # Default for uncategorized
            }

        # Base scenario: 20% market decline
        sp500_scenario = -0.20

        # First run the normal scenario as a baseline
        normal_results, _ = self.run_market_stress_test([sp500_scenario])
        normal_impact = normal_results.iloc[0]["Portfolio_Impact_EUR"]

        # Apply liquidity haircuts by category
        liquidity_impacts = []
        category_impacts = {}

        for idx, asset in self.portfolio_df.iterrows():
            value = asset["Valore in EUR"]
            category = asset["Category"]

            # Get the liquidity haircut for this category
            haircut = liquidity_haircuts.get(category, liquidity_haircuts["Other"])

            # Calculate the liquidity impact (additional loss)
            liquidity_impact = -value * haircut
            liquidity_impacts.append(liquidity_impact)

            # Aggregate by category
            if category not in category_impacts:
                category_impacts[category] = 0
            category_impacts[category] += liquidity_impact

        # Calculate total liquidity impact
        total_liquidity_impact = sum(liquidity_impacts)
        liquidity_pct_impact = total_liquidity_impact / self.total_value

        # Combined impact (market + liquidity)
        combined_impact = normal_impact + total_liquidity_impact
        combined_pct_impact = combined_impact / self.total_value

        # Prepare results
        results = {
            "Market_Only": {
                "SP500_Return": sp500_scenario,
                "Impact_Type": "Market Only",
                "Impact_EUR": normal_impact,
                "Impact_Pct": normal_impact / self.total_value,
                "New_Value": self.total_value + normal_impact
            },
            "Liquidity_Only": {
                "SP500_Return": 0,  # Liquidity impact only
                "Impact_Type": "Liquidity Only",
                "Impact_EUR": total_liquidity_impact,
                "Impact_Pct": liquidity_pct_impact,
                "New_Value": self.total_value + total_liquidity_impact
            },
            "Combined": {
                "SP500_Return": sp500_scenario,
                "Impact_Type": "Market + Liquidity",
                "Impact_EUR": combined_impact,
                "Impact_Pct": combined_pct_impact,
                "New_Value": self.total_value + combined_impact
            }
        }

        # Convert results to DataFrame
        results_df = pd.DataFrame(results).T
        category_impacts_df = pd.Series(category_impacts, name="Liquidity_Impact")

        return results_df, category_impacts_df


def main():
    """
    Main function to run stress tests on the portfolio.
    """
    # Load portfolio data
    try:
        portfolio_df = pd.read_csv("Portfolio.csv")
    except FileNotFoundError:
        # Sample data from the document
        data = """Prodotto,Codice,Quantità,Ultimo,Valore,Valore in EUR
CASH & CASH FUND & FTX CASH (EUR),,,,EUR 343.69,"343,69"
ISHARES $ TREASURY BOND 1-3YR U...,IE00B14X4S71,82,"115,53",EUR 9473.46,"9473,46"
ADR ON ADAPTIMMUNE THERAPEUTICS PLC,US00653A1079,3380,"0,23",USD 788.55,"720,19"
ALPHABET INC. - CLASS A,US02079K3059,20,"158,71",USD 3174.20,"2899,00"
ASML HOLDING,NL0010273215,10,"554,30",EUR 5543.00,"5543,00"
AETERNA ZENTARIS INC - NON TRADE...,CA22112H1192,10,"0,00",USD 0.00,"0,00"
AIRBUS SE,NL0000235190,38,"149,50",EUR 5681.00,"5681,00"
BUMBLE INC-A,US12047B1052,47,"4,16",USD 195.52,"178,57"
COSCIENS BIOPHARMA INC.  - COMMO...,CA22112H1010,22,"2,82",USD 62.04,"56,66"
DIASORIN,IT0003492391,20,"90,72",EUR 1814.40,"1814,40"
DIASORIN - NON TRADEABLE,IT0005643561,20,"0,00",EUR 0.00,"0,00"
DSV A/S,DK0060079531,40,"1055,00",DKK 42200.00,"5650,58"
GALAPAGOS,BE0003818359,18,"21,24",EUR 382.32,"382,32"
INTEL CORPORATION - CO,US4581401001,125,"21,53",USD 2691.25,"2457,92"
INFINEON TECHNOLOGIES AG,DE0006231004,101,"27,68",EUR 2795.68,"2795,68"
INVESCO US TREASURY BOND 7-10 YE...,IE00BF2FN646,100,"31,85",EUR 3185.00,"3185,00"
JUVENTUS FOOTBALL CLUB S.P.A,IT0005572778,361,"2,74",EUR 989.86,"989,86"
KERING,FR0000121485,17,"156,92",EUR 2667.64,"2667,64"
MONCLER,IT0004965148,60,"50,50",EUR 3030.00,"3030,00"
SMILEDIRECTCLUB INC. - COMMON STOCK,US83192H1068,400,"0,00",USD 0.04,"0,04"
SNDL INC,CA83307B1013,230,"1,39",USD 319.70,"291,98"
THE KRAFT HEINZ COMPAN,US5007541064,125,"29,10",USD 3637.50,"3322,13"
VANECK URANIUM AND NUCLEAR TECHN...,IE000M7V94E1,200,"22,25",EUR 4450.00,"4450,00"
VANGUARD FTSE ALL-WORLD UCITS ET...,IE00BK5BQT80,35,"110,50",EUR 3867.50,"3867,50"
WALT DISNEY COMPANY (T,US2546871060,35,"91,44",USD 3200.40,"2922,93"
XTRACKERS MSCI WORLD HEALTH CARE...,IE00BM67HK77,160,"43,17",EUR 6907.20,"6907,20"
ZOOM VIDEO COMMUNICATIONS-A,US98980L1017,50,"71,83",USD 3591.50,"3280,12"
ISHARES GLOBAL WATER UCITS ETF U...,IE00B1TXK627,85,"60,99",EUR 5184.15,"5184,15"
ISHARES MSCI EM ASIA UCITS ETF U...,IE00B5L8K969,36,"145,80",EUR 5248.80,"5248,80"
ISHARES MSCI SAUDI ARABIA CAPPED...,IE00BYYR0489,950,"5,42",EUR 5145.20,"5145,20"
"""
        import io
        portfolio_df = pd.read_csv(io.StringIO(data))

    # Initialize stress test
    stress_tester = StressTest(portfolio_df)

    # Run standard market stress test
    print("\n=== Standard Market Stress Test ===")
    results_df, category_impacts_df = stress_tester.run_market_stress_test()
    print(results_df)

    # Plot the results
    stress_tester.plot_stress_test_results(results_df, category_impacts_df)

    # Run correlation break stress test
    print("\n=== Correlation Break Stress Test ===")
    corr_results = stress_tester.run_correlation_break_stress_test()
    print(corr_results)

    # Run liquidity stress test
    print("\n=== Liquidity Stress Test ===")
    liquidity_results, liquidity_category = stress_tester.run_liquidity_stress_test()
    print(liquidity_results)
    print("\nLiquidity Impact by Category:")
    print(liquidity_category.sort_values())

    print("\nStress tests completed successfully.")

    return {
        "standard_stress": results_df,
        "category_impacts": category_impacts_df,
        "correlation_stress": corr_results,
        "liquidity_stress": liquidity_results,
        "liquidity_by_category": liquidity_category
    }


if __name__ == "__main__":
    main()