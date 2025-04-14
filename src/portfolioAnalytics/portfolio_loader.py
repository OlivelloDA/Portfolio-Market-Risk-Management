# portfolio_loader.py
# Class for loading and preprocessing portfolio data at 10/04/2025
# COPYRIGHT CLAUDIO OLIVELLI

import pandas as pd

class PortfolioLoader:
    """
    Class for loading and preprocessing portfolio data from a CSV file.

    This class handles:
    - Reading the portfolio CSV file
    - Cleaning and preprocessing the data
    - Calculating portfolio statistics
    - Converting values to a common currency (EUR)
    """

    def __init__(self, file_path):
        """
        Initialize the PortfolioLoader with the path to the portfolio CSV file.

        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing portfolio data
        """
        self.file_path = file_path

    def load_portfolio(self):
        """
        Load and preprocess the portfolio data.

        Returns:
        --------
        dict
            Dictionary containing:
            - 'data': Processed portfolio DataFrame
            - 'total_value': Total portfolio value in EUR
            - 'num_assets': Number of assets in the portfolio
            - 'currency_exposure': Dictionary with currency exposures
        """
        # Read the CSV file
        df = pd.read_csv(self.file_path)

        # Clean up column names
        df.columns = [col.strip() for col in df.columns]

        # Convert comma-separated values to dots for numerical processing
        # and convert string representations to numerical values
        df['Valore in EUR'] = df['Valore in EUR'].str.replace('.', '').str.replace(',', '.').astype(float)

        # Extract product types
        df['Product_Type'] = self._categorize_products(df['Prodotto'])

        # Calculate portfolio statistics
        total_value = df['Valore in EUR'].sum()
        num_assets = len(df[~df['Prodotto'].str.contains('CASH', case=False, na=False)])

        # Get currency exposures
        currency_exposure = self._calculate_currency_exposure(df)

        # Add weight column (% of portfolio)
        df['Weight'] = df['Valore in EUR'] / total_value

        # Remove non-tradeable items for further analysis
        analysis_df = df[~df['Prodotto'].str.contains('NON TRADE', case=False, na=False)].copy()

        return {
            'data': analysis_df,
            'total_value': total_value,
            'num_assets': num_assets,
            'currency_exposure': currency_exposure
        }

    def _categorize_products(self, product_series):
        """
        Categorize products into broader asset classes.

        Parameters:
        -----------
        product_series : pandas.Series
            Series containing product names

        Returns:
        --------
        pandas.Series
            Series with categorized product types
        """
        categories = []

        for product in product_series:
            if pd.isna(product):
                categories.append('Cash')
            elif 'CASH' in product.upper():
                categories.append('Cash')
            elif any(
                    etf in product.upper() for etf in ['ETF', 'ISHARES', 'VANGUARD', 'XTRACKERS', 'INVESCO', 'VANECK']):
                categories.append('ETF')
            elif any(sector in product.upper() for sector in ['HEALTH', 'WATER', 'TREASURY', 'URANIUM']):
                categories.append('Sector ETF')
            else:
                categories.append('Individual Stock')

        return pd.Series(categories)

    def _calculate_currency_exposure(self, df):
        """
        Calculate currency exposure of the portfolio.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing portfolio data

        Returns:
        --------
        dict
            Dictionary with currency exposures (amount and percentage)
        """
        # Extract currency from the 'Valore' column which has format like 'EUR 343.69'
        # For simplicity, we'll infer from the 'Valore' column where available

        # Create a currency column based on the beginning of 'Valore' column
        currency_exposure = {}

        # Most values are already in EUR as shown in 'Valore in EUR'
        eur_exposure = df['Valore in EUR'].sum()

        # For a more detailed breakdown, we would need more data on original currencies
        # This is a simplified implementation
        currency_exposure['EUR'] = {
            'amount': eur_exposure,
            'percentage': 100.0
        }

        return currency_exposure