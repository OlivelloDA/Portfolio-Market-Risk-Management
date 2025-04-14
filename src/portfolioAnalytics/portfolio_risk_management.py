# risk_manager.py
# Class for risk management analysis (VaR, Expected Shortfall, etc.)

import pandas as pd
import numpy as np
from scipy import stats


class RiskManager:
    """
    Class for performing risk management analysis on the portfolio.

    This class implements:
    - Value at Risk (VaR) calculation using different methods
    - Expected Shortfall (ES) calculation
    - Risk metrics for different time horizons
    - Risk decomposition by asset
    """

    def __init__(self, portfolio_data):
        """
        Initialize the RiskManager with portfolio data.

        Parameters:
        -----------
        portfolio_data : pandas.DataFrame
            DataFrame containing processed portfolio data
        """
        self.portfolio_data = portfolio_data
        self.total_value = portfolio_data['Valore in EUR'].sum()

        # Asset return assumptions
        # In a real implementation, these would be estimated from historical data
        self.return_assumptions = self._set_return_assumptions()

        # Asset correlation matrix
        # In a real implementation, this would be estimated from historical data
        self.correlation_matrix = self._set_correlation_matrix()

        # Calculate portfolio volatility
        self.portfolio_volatility = self._calculate_portfolio_volatility()

    def calculate_risk_metrics(self, confidence_level=0.99, time_horizon=1, method='parametric'):
        """
        Calculate Value at Risk (VaR) and Expected Shortfall (ES) for the portfolio.

        Parameters:
        -----------
        confidence_level : float, default 0.99
            Confidence level for VaR calculation (e.g., 0.95 for 95% VaR)
        time_horizon : int, default 1
            Time horizon in days
        method : str, default 'parametric'
            Method for VaR calculation:
            - 'parametric': Assumes normal distribution
            - 'historical': Uses historical simulation (not implemented here)
            - 'monte_carlo': Uses Monte Carlo simulation (not implemented here)

        Returns:
        --------
        tuple
            (VaR, ES) at the specified confidence level and time horizon

        Notes:
        ------
        Value at Risk (VaR) measures the maximum expected loss over a given time horizon
        at a specified confidence level under normal market conditions.

        Expected Shortfall (ES), also known as Conditional VaR (CVaR), measures the expected
        loss beyond the VaR threshold. It provides a more coherent risk measure than VaR.
        """
        if method == 'parametric':
            return self._parametric_var_es(confidence_level, time_horizon)
        elif method == 'historical':
            # Not implemented in this example
            print("Historical VaR not implemented")
            return None, None
        elif method == 'monte_carlo':
            # Not implemented in this example
            print("Monte Carlo VaR not implemented")
            return None, None
        else:
            raise ValueError(f"Unknown method: {method}")

    def _parametric_var_es(self, confidence_level, time_horizon):
        """
        Calculate parametric VaR and ES assuming normal distribution.

        Parameters:
        -----------
        confidence_level : float
            Confidence level for VaR calculation
        time_horizon : int
            Time horizon in days

        Returns:
        --------
        tuple
            (VaR, ES) at the specified confidence level and time horizon

        Notes:
        ------
        Parametric VaR assumes returns follow a normal distribution.
        The formula is: VaR = -μ*t + σ*√t*z_α
        where:
        - μ is the expected return
        - σ is the portfolio volatility
        - t is the time horizon
        - z_α is the z-score at the confidence level

        ES is calculated as: ES = -μ*t + σ*√t*φ(z_α)/[1-α]
        where φ is the standard normal PDF and α is the confidence level.
        """
        # Get z-score for the confidence level
        z_score = stats.norm.ppf(confidence_level)

        # Time scaling factor (square root of time rule)
        sqrt_t = np.sqrt(time_horizon)

        # Calculate portfolio expected return (annualized)
        # In a real implementation, this would be estimated from historical data or forecasts
        # Calculate portfolio expected return (annualized)
        # In a real implementation, this would be estimated from historical data or forecasts
        expected_return = 0.06  # Assuming 6% annualized expected return

        # Daily expected return
        daily_expected_return = expected_return * (time_horizon / 252)

        # Calculate VaR
        # VaR = -μ*t + σ*√t*z_α
        var = -daily_expected_return * self.total_value + self.portfolio_volatility * self.total_value * sqrt_t * z_score

        # Calculate ES (Expected Shortfall or Conditional VaR)
        # ES = -μ*t + σ*√t*φ(z_α)/[1-α]
        # where φ is the standard normal PDF
        phi_z = stats.norm.pdf(z_score)
        es = -daily_expected_return * self.total_value + self.portfolio_volatility * self.total_value * sqrt_t * (
                    phi_z / (1 - confidence_level))

        return var, es

    def _set_return_assumptions(self):
        """
        Set return and volatility assumptions for different asset types.

        Returns:
        --------
        dict
            Dictionary with return assumptions for different asset types

        Notes:
        ------
        These are simplified assumptions for demonstration purposes.
        In a real implementation, these would be estimated from historical data
        or forward-looking market expectations.
        """
        # Define annual return and volatility assumptions by asset type
        # Returns are in decimal form (e.g., 0.10 = 10%)
        return {
            'Cash': {'return': 0.02, 'volatility': 0.005},
            'Treasury': {'return': 0.04, 'volatility': 0.03},
            'Technology': {'return': 0.12, 'volatility': 0.25},
            'Healthcare': {'return': 0.08, 'volatility': 0.18},
            'Consumer': {'return': 0.07, 'volatility': 0.16},
            'Industrial': {'return': 0.09, 'volatility': 0.20},
            'Luxury': {'return': 0.11, 'volatility': 0.22},
            'Energy': {'return': 0.10, 'volatility': 0.24},
            'ETF': {'return': 0.08, 'volatility': 0.15},
            'Water': {'return': 0.07, 'volatility': 0.14},
            'Emerging Markets': {'return': 0.10, 'volatility': 0.22}
        }

    def _set_correlation_matrix(self):
        """
        Set correlation matrix for different asset types.

        Returns:
        --------
        pandas.DataFrame
            Correlation matrix for different asset types

        Notes:
        ------
        This is a simplified correlation matrix for demonstration purposes.
        In a real implementation, this would be estimated from historical data.
        """
        # Define the asset types we're using
        asset_types = [
            'Cash', 'Treasury', 'Technology', 'Healthcare', 'Consumer',
            'Industrial', 'Luxury', 'Energy', 'ETF', 'Water', 'Emerging Markets'
        ]

        # Create an empty correlation matrix
        corr_matrix = pd.DataFrame(index=asset_types, columns=asset_types)

        # Set diagonal to 1.0 (perfect self-correlation)
        np.fill_diagonal(corr_matrix.values, 1.0)

        # Set correlations between different asset types
        # These are simplified assumptions for demonstration purposes

        # Cash correlations
        corr_matrix.loc['Cash', 'Treasury'] = 0.2
        corr_matrix.loc[
            'Cash', ['Technology', 'Healthcare', 'Consumer', 'Industrial', 'Luxury', 'Energy', 'ETF', 'Water',
                     'Emerging Markets']] = 0.0

        # Treasury correlations
        corr_matrix.loc['Treasury', 'Cash'] = 0.2
        corr_matrix.loc['Treasury', ['Technology', 'Healthcare', 'Consumer', 'Industrial', 'Luxury', 'Energy', 'ETF',
                                     'Water']] = -0.1
        corr_matrix.loc['Treasury', 'Emerging Markets'] = -0.2

        # Technology correlations
        corr_matrix.loc['Technology', ['Cash', 'Treasury']] = 0.0
        corr_matrix.loc['Technology', 'Healthcare'] = 0.6
        corr_matrix.loc['Technology', 'Consumer'] = 0.7
        corr_matrix.loc['Technology', 'Industrial'] = 0.5
        corr_matrix.loc['Technology', 'Luxury'] = 0.4
        corr_matrix.loc['Technology', 'Energy'] = 0.3
        corr_matrix.loc['Technology', 'ETF'] = 0.8
        corr_matrix.loc['Technology', 'Water'] = 0.4
        corr_matrix.loc['Technology', 'Emerging Markets'] = 0.6

        # Healthcare correlations
        corr_matrix.loc['Healthcare', ['Cash', 'Treasury']] = 0.0
        corr_matrix.loc['Healthcare', 'Technology'] = 0.6
        corr_matrix.loc['Healthcare', 'Consumer'] = 0.5
        corr_matrix.loc['Healthcare', 'Industrial'] = 0.4
        corr_matrix.loc['Healthcare', 'Luxury'] = 0.3
        corr_matrix.loc['Healthcare', 'Energy'] = 0.2
        corr_matrix.loc['Healthcare', 'ETF'] = 0.7
        corr_matrix.loc['Healthcare', 'Water'] = 0.5
        corr_matrix.loc['Healthcare', 'Emerging Markets'] = 0.4

        # Consumer correlations
        corr_matrix.loc['Consumer', ['Cash', 'Treasury']] = 0.0
        corr_matrix.loc['Consumer', 'Technology'] = 0.7
        corr_matrix.loc['Consumer', 'Healthcare'] = 0.5
        corr_matrix.loc['Consumer', 'Industrial'] = 0.6
        corr_matrix.loc['Consumer', 'Luxury'] = 0.8
        corr_matrix.loc['Consumer', 'Energy'] = 0.4
        corr_matrix.loc['Consumer', 'ETF'] = 0.7
        corr_matrix.loc['Consumer', 'Water'] = 0.5
        corr_matrix.loc['Consumer', 'Emerging Markets'] = 0.6

        # Fill the remaining correlations with reasonable values
        # In a real implementation, these would be estimated from historical data

        # Industrial correlations
        corr_matrix.loc['Industrial', ['Cash', 'Treasury']] = 0.0
        corr_matrix.loc['Industrial', 'Technology'] = 0.5
        corr_matrix.loc['Industrial', 'Healthcare'] = 0.4
        corr_matrix.loc['Industrial', 'Consumer'] = 0.6
        corr_matrix.loc['Industrial', 'Luxury'] = 0.5
        corr_matrix.loc['Industrial', 'Energy'] = 0.7
        corr_matrix.loc['Industrial', 'ETF'] = 0.8
        corr_matrix.loc['Industrial', 'Water'] = 0.6
        corr_matrix.loc['Industrial', 'Emerging Markets'] = 0.5

        # Make the matrix symmetric
        for i in range(len(asset_types)):
            for j in range(i + 1, len(asset_types)):
                corr_matrix.iloc[j, i] = corr_matrix.iloc[i, j]

        return corr_matrix

    def _calculate_portfolio_volatility(self):
        """
        Calculate the portfolio's volatility.

        Returns:
        --------
        float
            Annualized portfolio volatility

        Notes:
        ------
        Portfolio volatility is calculated using the formula:
        σ_p = √(wᵀΣw)
        where:
        - w is the vector of asset weights
        - Σ is the covariance matrix of asset returns

        This implementation uses a simplified approach based on asset types.
        In a real implementation, this would use historical return data for all assets.
        """
        # Map assets to asset types
        asset_type_map = self._map_assets_to_types()

        # Calculate weights by asset type
        asset_type_weights = {}
        for _, row in self.portfolio_data.iterrows():
            asset_type = asset_type_map.get(row['Prodotto'], 'ETF')  # Default to ETF if not found
            weight = row['Weight']

            if asset_type in asset_type_weights:
                asset_type_weights[asset_type] += weight
            else:
                asset_type_weights[asset_type] = weight

        # Convert to lists for matrix operations
        types_list = list(asset_type_weights.keys())
        weights = np.array([asset_type_weights[t] for t in types_list])

        # Create a covariance matrix from correlations and volatilities
        volatilities = np.array([self.return_assumptions[t]['volatility'] for t in types_list])
        corr_submatrix = self.correlation_matrix.loc[types_list, types_list].values

        # Calculate the covariance matrix
        vol_matrix = np.diag(volatilities)
        cov_matrix = vol_matrix @ corr_submatrix @ vol_matrix

        # Calculate portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights

        # Take square root to get volatility
        portfolio_volatility = np.sqrt(portfolio_variance)

        return portfolio_volatility

    def _map_assets_to_types(self):
        """
        Map individual assets to asset types.

        Returns:
        --------
        dict
            Dictionary mapping asset names to asset types
        """
        # Define mappings based on keywords in asset names
        mapping = {}

        for _, row in self.portfolio_data.iterrows():
            product = str(row['Prodotto']).upper()

            if pd.isna(product) or 'CASH' in product:
                mapping[row['Prodotto']] = 'Cash'
            elif 'TREASURY' in product or 'BOND' in product:
                mapping[row['Prodotto']] = 'Treasury'
            elif any(keyword in product for keyword in ['ALPHABET', 'INTEL', 'INFINEON', 'ASML', 'ZOOM']):
                mapping[row['Prodotto']] = 'Technology'
            elif any(keyword in product for keyword in
                     ['ADAPTIMMUNE', 'DIASORIN', 'GALAPAGOS', 'HEALTH', 'AETERNA', 'COSCIENS']):
                mapping[row['Prodotto']] = 'Healthcare'
            elif any(keyword in product for keyword in ['DISNEY', 'KRAFT', 'BUMBLE']):
                mapping[row['Prodotto']] = 'Consumer'
            elif any(keyword in product for keyword in ['AIRBUS', 'DSV']):
                mapping[row['Prodotto']] = 'Industrial'
            elif any(keyword in product for keyword in ['KERING', 'MONCLER']):
                mapping[row['Prodotto']] = 'Luxury'
            elif 'URANIUM' in product:
                mapping[row['Prodotto']] = 'Energy'
            elif 'WATER' in product:
                mapping[row['Prodotto']] = 'Water'
            elif any(keyword in product for keyword in ['EM ASIA', 'SAUDI']):
                mapping[row['Prodotto']] = 'Emerging Markets'
            elif row['Product_Type'] in ['ETF', 'Sector ETF']:
                mapping[row['Prodotto']] = 'ETF'
            else:
                mapping[row['Prodotto']] = 'ETF'  # Default to ETF if not recognized

        return mapping

    def risk_decomposition(self):
        """
        Decompose portfolio risk by asset.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing risk contribution of each asset

        Notes:
        ------
        Risk contribution of asset i = w_i * (Σw)_i / σ_p
        where:
        - w_i is the weight of asset i
        - (Σw)_i is the ith element of the product of the covariance matrix and weights
        - σ_p is the portfolio volatility

        This shows how much each asset contributes to the total portfolio risk.
        """
        # Not fully implemented in this example
        # In a real implementation, this would calculate marginal risk contributions

        # Map assets to asset types for simplification
        asset_type_map = self._map_assets_to_types()

        # Create a risk contribution DataFrame
        risk_contrib = pd.DataFrame({
            'Asset': self.portfolio_data['Prodotto'],
            'Weight': self.portfolio_data['Weight'],
            'Asset_Type': [asset_type_map.get(p, 'Other') for p in self.portfolio_data['Prodotto']],
            'Value': self.portfolio_data['Valore in EUR']
        })

        # Add volatility from the asset types
        risk_contrib['Individual_Volatility'] = risk_contrib['Asset_Type'].map(
            {k: v['volatility'] for k, v in self.return_assumptions.items()}
        )

        # Calculate standalone risk (weight * individual volatility)
        risk_contrib['Standalone_Risk'] = risk_contrib['Weight'] * risk_contrib['Individual_Volatility']

        # Calculate naive risk contribution (assuming no correlations)
        total_standalone_risk = risk_contrib['Standalone_Risk'].sum()
        risk_contrib['Risk_Contribution_Pct'] = risk_contrib['Standalone_Risk'] / total_standalone_risk

        return risk_contrib