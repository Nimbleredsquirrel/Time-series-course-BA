# Time Series Course

Homework solutions for a time series forecasting course. The assignments cover the full progression from baseline methods to production-grade ML models.

## Notebooks

**HW1 - Forecasting using Simple Methods and Exponential Smoothing (ETS)**
Baseline forecasting on a bank liquidity time series and ATM cash demand data. Covers data frequency inference, train/test splits, WAPE/SMAPE/MAE metrics, recursive/direct/DirRec strategies, and ETS model selection via cross-validation.

**HW2 - Forecasting with (S)ARIMA(X) Models**
SARIMA on US macroeconomic population data and SARIMAX on GDP with exogenous variables. Covers parameter selection from ACF/PACF, ADF/KPSS/Hurst tests, interval estimates, residual analysis, Box-Cox transformations, and comparison across pmdarima/statsforecast/statsmodels.

**HW4 - Analyzing Time Series with Complex Trends and Seasonality**
Prophet model on hourly AEP energy consumption data plus a personal dataset (Google traffic flow). Covers holiday effects, linear vs logistic trend, hyperparameter tuning, BATS/TBATS modeling, and reasoning about SETAR applicability.

**HW5 - Applying ML Methods to Time Series Forecasting**
Gradient boosting for time series using CatBoost, LightGBM, and XGBoost. Covers calendar and STL-based features, Optuna hyperparameter search, feature importance and selection, quantile regression for interval forecasts, and global model training on a personal dataset.

## Data

- data_hw_1.csv - bank liquidity inflows/outflows (2017-2020, 2h frequency)
- atm_hw_1.csv - ATM daily cash withdrawal volumes (2015-2017)
- hw_AEP.csv - American Electric Power hourly energy consumption (2005-2018)
- train.csv - Google traffic flow dataset (5-min intervals, 2016)

## Stack

Python, pandas, statsmodels, Prophet, CatBoost, LightGBM, XGBoost, Optuna, scikit-learn, plotly, matplotlib
