# factor_analyzer.py
# Class for performing factor analysis on the portfolio

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class FactorAnalyzer:
    """
    Class for performing factor analysis on the portfolio.

    This class implements:
    - Analysis of exposure to different market factors (style factors, sectors, etc.)
    - Principal Component Analysis (PCA) to identify hidden factors
    - Visualization of factor exposures
    """

    def __init__(self, portfolio_data):
        """
        Initialize the FactorAnalyzer with portfolio data.

        Parameters:
        -----------
        portfolio_data : pandas.DataFrame
            DataFrame containing processed portfolio data
        """
        self.portfolio_data = portfolio_data
        self.factor_exposures = None

        # Define common market factors
        self.market_factors = {
            'Market': self._market_beta,
            'Size': self._size_factor,
            'Value': self._value_factor,
            'Growth': self._growth_factor,
            'Momentum': self._momentum_factor,
            'Quality': self._quality_factor,
            'Volatility': self._volatility_factor,
            'Dividend': self._dividend_factor
        }

        # Define sector classifications
        self.sectors = {
            'Technology': ['ALPHABET', 'INTEL', 'INFINEON', 'ASML', 'ZOOM'],
            'Healthcare': ['ADAPTIMMUNE', 'DIASORIN', 'GALAPAGOS', 'HEALTH CARE', 'AETERNA', 'COSCIENS'],
            'Consumer': ['DISNEY', 'KRAFT HEINZ', 'MONCLER', 'BUMBLE'],
            'Industrial': ['AIRBUS', 'DSV'],
            'Luxury': ['KERING', 'MONCLER'],
            'Energy': ['URANIUM'],
            'Fixed Income': ['TREASURY'],
            'Water': ['WATER'],
            'Emerging Markets': ['EM ASIA', 'SAUDI ARABIA']
        }

    def analyze_factors(self):
        """
        Perform comprehensive factor analysis on the portfolio.

        This method:
        1. Analyzes exposure to common market factors
        2. Analyzes sector exposures
        3. Performs PCA to identify hidden factors

        Returns:
        --------
        dict
            Dictionary containing factor exposure results
        """
        # Initialize results dictionary
        self.factor_exposures = {
            'market_factors': {},
            'sector_exposures': {},
            'pca_factors': {}
        }

        # Analyze exposure to market factors
        for factor_name, factor_func in self.market_factors.items():
            self.factor_exposures['market_factors'][factor_name] = factor_func()

        # Analyze sector exposures
        self.factor_exposures['sector_exposures'] = self._analyze_sector_exposures()

        # Perform PCA to identify hidden factors
        self.factor_exposures['pca_factors'] = self._perform_pca()

        return self.factor_exposures

    def display_factor_exposures(self):
        """
        Display the factor exposures in a readable format.
        """
        if self.factor_exposures is None:
            print("Run analyze_factors() first to calculate factor exposures.")
            return

        print("\nMarket Factor Exposures:")
        for factor, exposure in self.factor_exposures['market_factors'].items():
            print(f"  {factor}: {exposure:.2f}")

        print("\nSector Exposures:")
        for sector, exposure in self.factor_exposures['sector_exposures'].items():
            print(f"  {sector}: {exposure * 100:.2f}%")

        print("\nPCA Factor Analysis:")
        for i, var in enumerate(self.factor_exposures['pca_factors']['explained_variance_ratio']):
            print(f"  Factor {i + 1}: {var * 100:.2f}% of variance explained")

    def _market_beta(self):
        """
        Calculate the portfolio's exposure to market beta.

        Returns:
        --------
        float
            Portfolio's estimated market beta

        Notes:
        ------
        Market beta measures the portfolio's sensitivity to market movements.
        Beta = 1: Portfolio moves with the market
        Beta > 1: Portfolio is more volatile than the market
        Beta < 1: Portfolio is less volatile than the market
        """
        # Assign approximate betas to different asset types
        # In a real implementation, these would be calculated using regression analysis
        # against market returns

        beta_approx = {
            'Individual Stock': 1.1,  # Average stock has slightly higher beta than market
            'ETF': 1.0,  # Broad market ETFs typically have beta close to 1
            'Sector ETF': 0.9,  # Sector ETFs vary but often defensive sectors have lower beta
            'Cash': 0.0  # Cash has zero beta
        }

        # Calculate weighted average beta
        weighted_beta = sum(
            self.portfolio_data['Weight'] *
            self.portfolio_data['Product_Type'].map(beta_approx).fillna(1.0)
        )

        return weighted_beta

    def _size_factor(self):
        """
        Calculate the portfolio's exposure to the size factor.

        Returns:
        --------
        float
            Portfolio's estimated size factor exposure (-1 to 1 scale)
            Positive values indicate large-cap bias, negative values indicate small-cap bias

        Notes:
        ------
        The size factor captures the tendency of smaller companies to outperform larger
        companies over long periods (small-cap premium).
        """
        # Approximate size factor exposure based on product types and names
        # In a real implementation, this would use actual market capitalizations

        large_cap_indicators = ['ALPHABET', 'DISNEY', 'ASML', 'KRAFT', 'INTEL', 'AIRBUS']
        mid_cap_indicators = ['INFINEON', 'MONCLER', 'KERING', 'DIASORIN', 'GALAPAGOS']
        small_cap_indicators = ['BUMBLE', 'ADAPTIMMUNE', 'AETERNA', 'COSCIENS', 'SNDL']

        # Score each holding based on company size indicators
        size_scores = []

        for _, row in self.portfolio_data.iterrows():
            product = str(row['Prodotto']).upper()

            if row['Product_Type'] == 'ETF' or row['Product_Type'] == 'Sector ETF':
                # Most ETFs are market-cap weighted, so assume large-cap bias
                size_scores.append(0.5)
            elif row['Product_Type'] == 'Cash':
                # Cash has no size exposure
                size_scores.append(0)
            elif any(indicator in product for indicator in large_cap_indicators):
                size_scores.append(0.8)  # Strong large-cap
            elif any(indicator in product for indicator in mid_cap_indicators):
                size_scores.append(0)  # Neutral
            elif any(indicator in product for indicator in small_cap_indicators):
                size_scores.append(-0.8)  # Strong small-cap
            else:
                size_scores.append(0)  # Unknown, assume neutral

        # Calculate weighted average size score
        self.portfolio_data['Size_Score'] = size_scores
        weighted_size_factor = sum(self.portfolio_data['Weight'] * self.portfolio_data['Size_Score'])

        return weighted_size_factor

    def _value_factor(self):
        """
        Calculate the portfolio's exposure to the value factor.

        Returns:
        --------
        float
            Portfolio's estimated value factor exposure (-1 to 1 scale)
            Positive values indicate value bias, negative values indicate growth bias

        Notes:
        ------
        The value factor captures the tendency of undervalued companies (low P/E, P/B, etc.)
        to outperform overvalued companies over long periods (value premium).
        """
        # Approximate value factor exposure based on product types and names
        # In a real implementation, this would use actual valuation metrics

        value_indicators = ['KRAFT', 'INTEL', 'TREASURY']
        growth_indicators = ['ALPHABET', 'ASML', 'ZOOM', 'ADAPTIMMUNE', 'GALAPAGOS']

        # Score each holding based on value/growth indicators
        value_scores = []

        for _, row in self.portfolio_data.iterrows():
            product = str(row['Prodotto']).upper()

            if 'TREASURY' in product or 'BOND' in product:
                # Fixed income typically has no value/growth exposure
                value_scores.append(0)
            elif row['Product_Type'] == 'Cash':
                # Cash has no value/growth exposure
                value_scores.append(0)
            elif any(indicator in product for indicator in value_indicators):
                value_scores.append(0.7)  # Strong value
            elif any(indicator in product for indicator in growth_indicators):
                value_scores.append(-0.7)  # Strong growth
            else:
                value_scores.append(0)  # Unknown, assume neutral

        # Calculate weighted average value score
        self.portfolio_data['Value_Score'] = value_scores
        weighted_value_factor = sum(self.portfolio_data['Weight'] * self.portfolio_data['Value_Score'])

        return weighted_value_factor

    def _growth_factor(self):
        """
        Calculate the portfolio's exposure to the growth factor.

        Returns:
        --------
        float
            Portfolio's estimated growth factor exposure (-1 to 1 scale)
            Positive values indicate high growth bias, negative values indicate low growth bias

        Notes:
        ------
        The growth factor measures exposure to companies with high expected growth rates.
        This is often the opposite of value, but not always.
        """
        # Growth is often the inverse of value, but not exactly
        # For simplicity, we'll estimate it as the negative of the value factor
        # In a real implementation, this would use growth metrics like sales growth, etc.

        return -self._value_factor()

    def _momentum_factor(self):
        """
        Calculate the portfolio's exposure to the momentum factor.

        Returns:
        --------
        float
            Portfolio's estimated momentum factor exposure (-1 to 1 scale)
            Positive values indicate high momentum bias

        Notes:
        ------
        The momentum factor captures the tendency of assets that have performed well
        in the recent past to continue performing well in the near future.
        """
        # Approximate momentum factor exposure
        # In a real implementation, this would use price momentum data

        # Companies with recent momentum (hypothetical for this example)
        high_momentum = ['ASML', 'ALPHABET', 'ZOOM', 'MONCLER']
        low_momentum = ['INTEL', 'BUMBLE', 'ADAPTIMMUNE', 'JUVENTUS']

        # Score each holding based on momentum indicators
        momentum_scores = []

        for _, row in self.portfolio_data.iterrows():
            product = str(row['Prodotto']).upper()

            if row['Product_Type'] == 'Cash':
                # Cash has no momentum exposure
                momentum_scores.append(0)
            elif any(company in product for company in high_momentum):
                momentum_scores.append(0.8)  # High momentum
            elif any(company in product for company in low_momentum):
                momentum_scores.append(-0.8)  # Low momentum
            else:
                momentum_scores.append(0)  # Unknown, assume neutral

        # Calculate weighted average momentum score
        self.portfolio_data['Momentum_Score'] = momentum_scores
        weighted_momentum_factor = sum(self.portfolio_data['Weight'] * self.portfolio_data['Momentum_Score'])

        return weighted_momentum_factor

    def _quality_factor(self):
        """
        Calculate the portfolio's exposure to the quality factor.

        Returns:
        --------
        float
            Portfolio's estimated quality factor exposure (-1 to 1 scale)
            Positive values indicate high quality bias

        Notes:
        ------
        The quality factor captures exposure to companies with strong balance sheets,
        high profitability, and stable earnings growth.
        """
        # Approximate quality factor exposure
        # In a real implementation, this would use financial metrics

        high_quality = ['ASML', 'ALPHABET', 'DISNEY', 'AIRBUS', 'KERING']
        low_quality = ['ADAPTIMMUNE', 'JUVENTUS', 'AETERNA', 'SMILEDIRECTCLUB']

        # Score each holding based on quality indicators
        quality_scores = []

        for _, row in self.portfolio_data.iterrows():
            product = str(row['Prodotto']).upper()

            if 'TREASURY' in product or 'BOND' in product:
                # Government bonds typically have high quality
                quality_scores.append(0.9)
            elif row['Product_Type'] == 'Cash':
                # Cash has highest quality
                quality_scores.append(1.0)
            elif any(company in product for company in high_quality):
                quality_scores.append(0.7)  # High quality
            elif any(company in product for company in low_quality):
                quality_scores.append(-0.7)  # Low quality
            else:
                quality_scores.append(0)  # Unknown, assume neutral

        # Calculate weighted average quality score
        self.portfolio_data['Quality_Score'] = quality_scores
        weighted_quality_factor = sum(self.portfolio_data['Weight'] * self.portfolio_data['Quality_Score'])

        return weighted_quality_factor

    def _volatility_factor(self):
        """
        Calculate the portfolio's exposure to the volatility factor.

        Returns:
        --------
        float
            Portfolio's estimated volatility factor exposure (-1 to 1 scale)
            Positive values indicate high volatility bias

        Notes:
        ------
        The volatility factor captures exposure to assets with high or low price volatility.
        Low volatility assets tend to outperform high volatility assets on a risk-adjusted basis.
        """
        # Approximate volatility factor exposure
        # In a real implementation, this would use historical volatility data

        high_volatility = ['BUMBLE', 'ADAPTIMMUNE', 'SNDL', 'JUVENTUS', 'URANIUM']
        low_volatility = ['TREASURY', 'BOND', 'KRAFT', 'DISNEY']

        # Score each holding based on volatility indicators
        volatility_scores = []

        for _, row in self.portfolio_data.iterrows():
            product = str(row['Prodotto']).upper()

            if 'TREASURY' in product or 'BOND' in product:
                # Fixed income typically has low volatility
                volatility_scores.append(-0.8)
            elif row['Product_Type'] == 'Cash':
                # Cash has lowest volatility
                volatility_scores.append(-1.0)
            elif any(indicator in product for indicator in high_volatility):
                volatility_scores.append(0.8)  # High volatility
            elif any(indicator in product for indicator in low_volatility):
                volatility_scores.append(-0.7)  # Low volatility
            else:
                volatility_scores.append(0)  # Unknown, assume neutral

        # Calculate weighted average volatility score
        self.portfolio_data['Volatility_Score'] = volatility_scores
        weighted_volatility_factor = sum(self.portfolio_data['Weight'] * self.portfolio_data['Volatility_Score'])

        return weighted_volatility_factor

    def _dividend_factor(self):
        """
        Calculate the portfolio's exposure to the dividend factor.

        Returns:
        --------
        float
            Portfolio's estimated dividend factor exposure (-1 to 1 scale)
            Positive values indicate high dividend bias

        Notes:
        ------
        The dividend factor captures exposure to companies with high dividend yields.
        """
        # Approximate dividend factor exposure
        # In a real implementation, this would use actual dividend yield data

        high_dividend = ['KRAFT', 'INTEL', 'WATER']
        low_dividend = ['ZOOM', 'ADAPTIMMUNE', 'BUMBLE', 'JUVENTUS']

        # Score each holding based on dividend indicators
        dividend_scores = []

        for _, row in self.portfolio_data.iterrows():
            product = str(row['Prodotto']).upper()

            if 'TREASURY' in product or 'BOND' in product:
                # Fixed income has yield but not dividends
                dividend_scores.append(0.5)
            elif row['Product_Type'] == 'Cash':
                # Cash might have interest but not dividends
                dividend_scores.append(0)
            elif any(indicator in product for indicator in high_dividend):
                dividend_scores.append(0.8)  # High dividend
            elif any(indicator in product for indicator in low_dividend):
                dividend_scores.append(-0.8)  # Low/no dividend
            else:
                dividend_scores.append(0)  # Unknown, assume neutral

        # Calculate weighted average dividend score
        self.portfolio_data['Dividend_Score'] = dividend_scores
        weighted_dividend_factor = sum(self.portfolio_data['Weight'] * self.portfolio_data['Dividend_Score'])

        return weighted_dividend_factor

    def _analyze_sector_exposures(self):
        """
        Analyze the portfolio's exposure to different sectors.

        Returns:
        --------
        dict
            Dictionary containing sector exposures (as percentage of portfolio)
        """
        sector_exposures = {sector: 0.0 for sector in self.sectors}

        # Calculate exposure to each sector
        for _, row in self.portfolio_data.iterrows():
            product = str(row['Prodotto']).upper()
            weight = row['Weight']

            # Check which sector(s) the product belongs to
            matched_sectors = []
            for sector, keywords in self.sectors.items():
                if any(keyword.upper() in product for keyword in keywords):
                    matched_sectors.append(sector)

            if matched_sectors:
                # Distribute weight equally among matched sectors
                sector_weight = weight / len(matched_sectors)
                for sector in matched_sectors:
                    sector_exposures[sector] += sector_weight

        return sector_exposures

    def _perform_pca(self):
        """
        Perform Principal Component Analysis to identify hidden factors.

        Returns:
        --------
        dict
            Dictionary containing PCA results

        Notes:
        ------
        PCA identifies orthogonal factors that explain the variance in the portfolio.
        The first principal component typically represents market exposure.
        Subsequent components may represent sector, style, or other factor exposures.
        """
        # In a real implementation, this would use return data for PCA
        # For this example, we'll use the factor scores as a proxy

        # Collect all factor scores
        factor_scores = self.portfolio_data[[
            'Size_Score', 'Value_Score', 'Momentum_Score',
            'Quality_Score', 'Volatility_Score', 'Dividend_Score'
        ]].fillna(0)

        # Perform PCA
        # In a real implementation, we would use more data points (daily returns)
        pca = PCA()
        pca.fit(factor_scores)

        # Return PCA results
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }