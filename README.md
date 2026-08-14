# Forecasting Technology and Energy Stocks Through Macroeconomic Disruption

Investigated whether macroeconomic and policy signals, inflation, Treasury yields, tariff events, and consumer sentiment, improve stock price forecasting beyond price history alone, comparing tech and energy sector stocks using a three-model ladder (ARIMA, Linear Regression, CNN-LSTM). Applied time-series modeling, feature engineering, and cross-model evaluation design in Python, all within AI4ALL's AI4ALL Ignite accelerator.

## Problem Statement

Retail investors typically lack access to the macro-aware forecasting tools available to institutional players, and it's unclear how much macroeconomic and policy data actually improves stock price prediction over price history alone. This matters for two reasons: first, understanding whether macro signals genuinely add predictive value shapes what kind of tools are worth building for everyday investors; second, a forecasting tool that appears predictive without being rigorously tested against a naive baseline risks giving users false confidence and encouraging risky decisions based on noise rather than signal. This project tests that question directly, using tech and energy sector stocks as a comparison case, since the two sectors respond differently to macroeconomic disruption.

## Key Results

1. Cleaned and engineered a dataset of 1,254 trading days (July 2021–July 2026) across 14 tickers (7 tech, 7 energy), 42 raw features, plus engineered technical indicators and macroeconomic variables
2. Found that daily stock prices behave like a random walk: ARIMA, using price history alone, did not outperform a naive "no change" forecast for any of the 14 tickers
   * Forecast errors clustered around real volatility spikes (April 2025 and June 2026 tariff shocks)
   * Low-priced, low-volatility tickers (e.g., ET) converged near-perfectly with the naive baseline; high-priced, high-volatility tickers (e.g., META) showed the widest absolute error
3. Built a baseline-vs-full comparison in the regression model to directly test macro feature value, finding that adding macro and cross-ticker features increased average RMSE rather than decreasing it, likely due to multicollinearity and missing-data imputation across macro columns, a result that argues against assuming more features are automatically better
4. Identified and corrected multiple methodological issues that would have otherwise inflated reported accuracy: a one-shot ARIMA forecast that compounded error over time (fixed with a rolling forecast), a shuffled train/test split causing data leakage in the regression model, and a flattened multi-scale RMSE calculation that produced a meaningless error metric in the LSTM model
5. Identified next steps for closing the gap ARIMA's architecture leaves open: adding SARIMAX (ARIMA with exogenous macro regressors), standardizing evaluation metrics and forecast horizons across all three models, and extracting feature importance to identify which macro factors matter most

## Methodologies

We designed a three-model ladder to isolate the contribution of price history, macroeconomic data, and deep learning respectively. ARIMA served as a univariate baseline, forecasting each ticker's price using only its own history, with order selection automated via `auto_arima` and evaluated using a rolling forecast (refit at each step with the newly observed actual, rather than a one-shot multi-step forecast, which we found compounded error significantly). Linear Regression compared a baseline feature set (a ticker's own price and volume) against a full feature set (baseline plus macroeconomic variables and other tickers' data) to directly test whether macro signals improved prediction. CNN-LSTM used a sequence-to-sequence architecture (Conv1D encoder, LSTM encoder-decoder) to forecast across all 42 features simultaneously from a 126-day lookback window.

Data was sourced from a market data provider (14 tickers' daily Close/Volume) and merged with macroeconomic series from FRED (CPI, Consumer Sentiment), U.S. Treasury daily yield curve data, and a manually curated tariff-event calendar. Cleaning involved stripping currency formatting from price fields, forward/back-filling sparse Treasury yield columns, and resampling monthly macro series (CPI, Consumer Sentiment) to daily frequency via forward-fill before merging. Engineered features included technical indicators (20/50-day SMA, 10-day momentum, 20-day rolling volatility, 14-day RSI) and a next-day log-return target. All models were evaluated using MAE, RMSE, and MAPE (to enable fair comparison across tickers at very different price scales), with ARIMA additionally benchmarked against a naive "no change" forecast to guard against overstating predictive accuracy.

## Data Sources

* Market data provider: daily historical OHLCV data for 14 tickers (2021–2026)
* [FRED: Consumer Price Index (CPIAUCSL)](https://fred.stlouisfed.org/series/CPIAUCSL)
* [FRED: University of Michigan Consumer Sentiment (UMCSENT)](https://fred.stlouisfed.org/series/UMCSENT)
* [U.S. Treasury: Daily Par Yield Curve Rates](https://home.treasury.gov/resource-center/data-chart-center/interest-rates)
* [USTR: Trade Policy Press Releases](https://ustr.gov/about-us/policy-offices/press-office/press-releases) (used to construct the tariff event calendar)

## Technologies Used

* Python
* pandas, NumPy
* statsmodels, pmdarima (ARIMA / auto_arima)
* scikit-learn (Linear Regression, train/test splitting, evaluation metrics)
* TensorFlow / Keras (CNN-LSTM)
* pandas-datareader (FRED data access)
* matplotlib, seaborn (visualization)

## Authors

This project was completed in collaboration with:
Abby Lei, Zuzu Cho Oo
