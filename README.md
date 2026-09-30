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

## Author

*Samuel Birhanu Abebe*

Data Science student interested in machine learning, forecasting, and applications of AI to Ethiopian coffee.
