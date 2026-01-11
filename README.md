# Comprehensive Stock Market Analysis

## Overview

This project demonstrates **quantitative analysis techniques** for educational purposes, exploring traditional statistical methods, time series modeling, and machine learning applied to stock market data.

**Important**: This is an **academic exercise** showing common analytical workflows, not a production trading system. The methods demonstrated here are:
- Useful for learning quantitative techniques and programming skills
- **Not representative of institutional quantitative finance**
- Likely to have limited or no predictive power in live markets

This project conducts a quantitative analysis of stock market dynamics using a combination of traditional statistical methods, time series modeling, and modern machine learning techniques. The analysis explores price movements, volatility patterns, risk metrics, and predictive relationships across a diverse portfolio of technology and energy stocks.

## Key Features

### Statistical & Time Series Analysis
- **ARIMA Modeling**: Autoregressive integrated moving average models for price forecasting
- **GARCH Volatility Modeling**: Generalized autoregressive conditional heteroskedasticity for volatility estimation
- **Stationarity Testing**: Augmented Dickey-Fuller tests

### Machine Learning
- **Random Forest Classifier**: Predicts next-day price direction using technical indicators
- **LSTM Neural Networks**: Deep learning for multi-step price forecasting 
- **Feature Engineering**: Technical indicators including RSI, MACD, Bollinger Bands, and momentum indicators

### Risk & Portfolio Analysis
- **Sharpe Ratio**: Risk-adjusted return calculations
- **Maximum Drawdown**: Worst peak-to-trough decline analysis
- **Monte Carlo Simulation**: Value-at-Risk (VaR) estimation using 10,000+ simulations
- **Correlation Analysis**: Cross-asset correlations for diversification insights

## Dataset

- **Stocks Analyzed**: AAPL, GOOG, MSFT, TSLA, XOM, CVX, COP, NVDA
- **Time Period**: 250 trading days (January 13, 2025 - January 9, 2026)
- **Price Data**: Closing prices (unadjusted)
  - Note: Dividends and stock splits are not adjusted for in this analysis
  - For production analysis, adjusted close prices would be more appropriate
- **Data Source**: Yahoo Finance via yfinance API
  - Convenient for educational purposes but may contain occasional gaps or adjustment quirks
  - For critical applications, consider validated commercial data providers

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Required Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies (for LSTM models)

```bash
pip install tensorflow
```

## Usage

### Running the Analysis

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook Enhanced_Stock_Analysis.ipynb
   ```

3. **Run all cells**: 
   - Click `Kernel` → `Restart & Run All`
   - Or run cells sequentially for step-by-step execution

### Customization

To analyze different stocks, modify the `stock_list` variable in the second code cell:

```python
stock_list = ['YOUR', 'STOCK', 'TICKERS', 'HERE']
```

## Project Structure

```
stock-analysis-project/
├── Enhanced_Stock_Analysis.ipynb    # Main analysis notebook
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── Stock_Market_Analysis_Project (1).ipynb  # Original notebook (archived)
```

## Key Findings

1. **Market Correlations**: Tech stocks demonstrate strong positive correlation (>0.85), forming a distinct cluster from energy stocks
2. **Predictive Power**: Random Forest achieves meaningful accuracy in predicting price direction, with RSI and MACD as top features
3. **Volatility Patterns**: GARCH models successfully identify volatility clustering, particularly pronounced in TSLA
4. **Risk-Return Profiles**: Significant variation in Sharpe ratios across stocks, informing portfolio allocation decisions
5. **VaR Estimation**: Monte Carlo simulations provide probabilistic forecasts with 95% confidence intervals

## Technologies Used

- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Statistical Modeling**: statsmodels, arch
- **Machine Learning**: scikit-learn
- **Deep Learning**: TensorFlow/Keras (optional)
- **Data Acquisition**: yfinance, pandas-datareader

## Methodology

The analysis follows a structured approach:

1. **Data Collection**: Historical stock data from Yahoo Finance
2. **Exploratory Analysis**: Price trends, volume, returns distribution
3. **Time Series Forecasting**: ARIMA models for short-term predictions
4. **Volatility Modeling**: GARCH for conditional heteroskedasticity
5. **Machine Learning**: Random Forest and LSTM for pattern recognition
6. **Risk Assessment**: Comprehensive risk metrics and VaR calculation
7. **Insights & Conclusions**: Actionable findings and limitations

## Limitations & Disclaimers

- **Not Financial Advice**: This analysis is for educational and research purposes only
- **Historical Performance**: Past performance does not guarantee future results
- **Model Limitations**: Predictions degrade significantly beyond short time horizons
- **Assumptions**: Models assume log-normal returns and may not capture black swan events
- **Transaction Costs**: Analysis does not account for trading fees and market impact

## Future Enhancements

- Extend the horizon to multi-year data to test stability across regimes.
- Run cost-aware backtests (transaction costs, slippage) to evaluate whether ML edges survive implementation.
- Explore ensemble signals (technical/ML + volatility regime information) and portfolio-level Monte Carlo VaR.

## Academic & Educational Context

This project demonstrates **common quantitative techniques for educational purposes**. The technical indicators (RSI, MACD, Bollinger Bands) used here are:

- **Standard in academic curricula** and widely taught in financial education
- **Useful for learning** feature engineering, ML workflows, and risk analysis
- **Not representative** of cutting-edge quantitative finance used by institutional firms

### What This Project Teaches

This analysis demonstrates valuable skills:
1. **Proper time series modeling** workflows (stationarity testing, ARIMA, GARCH)
2. **ML validation techniques** (walk-forward validation, avoiding look-ahead bias)
3. **Risk metric calculations** (Sharpe, Sortino, VaR, CVaR, drawdowns)
4. **The core concepts** underlying momentum, volatility, and mean reversion
5. **Data engineering** and visualization with financial data

### Real-World Quantitative Finance

**Note for practitioners**: Sophisticated quantitative firms typically use:
- **Market microstructure data**: Order flow, bid-ask dynamics, trade imbalances
- **Alternative data sources**: Satellite imagery, credit card transactions, shipping data
- **Proprietary features**: Derived from statistical/information theory, not textbook indicators  
- **Rigorous volatility estimators**: Beyond simple rolling windows or Bollinger Bands
- **Factor models and cross-sectional approaches**: Rather than individual stock prediction

The **technical indicators** in this project (RSI, MACD, Bollinger Bands) are:
- Simple transformations of price/volume data the model already has
- Widely known patterns that have been largely competed away
- Unlikely to provide significant predictive edge in live markets

This project's value is in demonstrating **methodological rigor** (proper train/test splits, walk-forward validation, honest evaluation), not in discovering a trading edge.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Data provided by Yahoo Finance
- Statistical methods based on classical quantitative finance literature

---

**Disclaimer**: This analysis is for educational purposes only. Always conduct your own research and consult with financial professionals before making investment decisions.

**License:** MIT

www.linkedin.com/in/willis-yorick/
