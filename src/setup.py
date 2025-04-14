from setuptools import setup, find_packages

setup(
    name="portfolio_analyzer",
    version="0.1.0",
    description="A comprehensive tool for portfolio analysis, risk management, and stress testing",
    author="Portfolio Analysis Team",
    author_email="example@example.com",  # Replace with your email
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "statsmodels" # For factor analysis
    ],
    python_requires='>=3.7',  # Specify minimum Python version
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)