# Sidama Coffee Price Predictor

# Project Overview

This project develops a time-series forecasting system for Ethiopian Sidama coffee export prices.

The project analyzes historical coffee price data and applies statistical and machine-learning methods to identify patterns and forecast future prices.

# Research Question

Can historical Sidama coffee export prices be used to forecast future prices using time-series and machine-learning methods?

# Objectives

- Analyze historical Sidama coffee price patterns
- Explore trends and seasonal behavior
- Create features from historical price observations
- Develop forecasting models
- Compare machine-learning forecasts with a baseline approach
- Evaluate forecasting performance using time-based validation

# Dataset

The dataset contains historical Sidama coffee price observations with information including:

- Date
- Coffee price in USD/lb
- Origin
- Grade
- Processing method
- Export type

The dataset covers observations from 2021 to 2026.

# Methodology

The project follows these main steps:

1. Data preparation
2. Exploratory time-series analysis
3. Feature engineering
4. Model development
5. Time-based model evaluation
6. Forecast analysis

# Feature Engineering

Historical price information is transformed into predictive features including:

- Lagged prices
- Rolling averages
- Rolling volatility
- Month
- Week
- Cyclical seasonal features

# Models

The project currently explores:

- Naive forecasting baseline
- Random Forest regression

Additional time-series models will be evaluated as the project develops.

# Evaluation

Model performance will be evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The evaluation uses chronological train/test splitting to avoid using future observations to predict the past.

# Project Status

This project is actively being developed. Future improvements include additional time-series models, walk-forward validation, improved visualizations, and deeper analysis of forecasting errors.

# Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib


## Dataset

The project uses historical Ethiopian Sidama coffee export price data.

The dataset contains weekly observations with the following variables:

| Variable | Description |
|---|---|
| `date` | Observation date |
| `price_usd_lb` | Coffee price in USD per pound |
| `origin` | Coffee origin |
| `grade` | Coffee grade |
| `process` | Processing method |
| `export_type` | Export category |
| `source` | Price data source |

The dataset covers observations from 2021 to 2026 and focuses on Sidama coffee under Grade 1 and Grade 2 natural commercial categories.

### Data Source

The price data are based on EEM / ECTA minimum price records included in the dataset.

### Target Variable

The primary forecasting target is:

`price_usd_lb`

which represents the coffee price in US dollars per pound.



## Methodology

The project follows a time-series forecasting workflow:

1. Load and inspect the historical coffee price data.
2. Convert the `date` variable into a datetime format.
3. Sort observations chronologically.
4. Explore historical price patterns and trends.
5. Prepare the time-series data for forecasting.
6. Train forecasting models using historical observations.
7. Evaluate predictions using appropriate forecasting metrics.
8. Generate forecasts for future coffee prices.

### Forecasting Approach

The project uses statistical time-series methods to model historical Sidama coffee prices.

The main forecasting approach is based on ARIMA (AutoRegressive Integrated Moving Average), which is designed to model temporal dependencies and trends in time-series data.

The project also includes machine-learning components for comparison and forecasting analysis.

### Train-Test Strategy

Because this is a time-series problem, observations are kept in chronological order.

Earlier observations are used for model training, while later observations are reserved for evaluating forecasting performance.

This avoids randomly shuffling the data and helps prevent information from the future from being used to predict the past.


## Model Evaluation

Forecasting performance is evaluated by comparing predicted prices with the observed prices in the test period.

The project uses the following evaluation metrics:

- **MAE (Mean Absolute Error):** measures the average absolute difference between predicted and observed prices.
- **RMSE (Root Mean Squared Error):** measures prediction error while giving greater weight to larger errors.
- **R² (Coefficient of Determination):** measures how much of the variation in the target variable is explained by the model.

These metrics provide complementary information about forecasting accuracy and model performance.


## Results

The forecasting pipeline evaluates model performance on observations that were not used during model training.

Model performance will be reported using MAE, RMSE, and R² after the forecasting pipeline has been fully validated.

Forecast visualizations will also be included to compare predicted and observed coffee prices over time.


## Author

*Samuel Birhanu Abebe*

Data Science student interested in machine learning, forecasting, and applications of AI to Ethiopian coffee.
