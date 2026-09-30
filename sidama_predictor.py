# Samuel Birhanu Abebe
# Sidama Coffee Price Predictor

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# DATA REQUIREMENTS
# ============================================================

REQUIRED_COLUMNS = [
    "date",
    "price_usd_lb",
    "origin",
    "grade",
    "process",
]

OPTIONAL_NUMERIC_COLUMNS = [
    "usd_etb",
    "arabica_cents_lb",
    "rainfall_mm",
    "temperature_c",
    "export_volume_kg",
]


# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

def load_data(csv_file):
    """Load, validate, clean, and filter the coffee price dataset."""

    data = pd.read_csv(csv_file)

    # Check required columns
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Your CSV is missing these columns: "
            + ", ".join(missing)
        )

    # Convert date and target to appropriate types
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data["price_usd_lb"] = pd.to_numeric(
        data["price_usd_lb"],
        errors="coerce",
    )

    # Convert optional numeric columns when available
    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    # Clean categorical variables
    data["origin"] = (
        data["origin"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data["grade"] = (
        data["grade"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    data["process"] = (
        data["process"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Remove rows without date or price
    data = data.dropna(
        subset=["date", "price_usd_lb"]
    )

    # Keep Sidama observations
    data = data[
        data["origin"] == "sidama"
    ].copy()

    if data.empty:
        raise ValueError(
            "No Sidama rows found. "
            "Use 'Sidama' in the origin column."
        )

    # Sort each coffee series chronologically
    data = data.sort_values(
        ["grade", "process", "date"]
    ).reset_index(drop=True)

    return data


# ============================================================
# EXPLORATORY VISUALIZATION
# ============================================================

def create_price_history_plot(data, output_folder):
    """Create a historical coffee price plot."""

    output_path = Path(output_folder)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 6))

    for (grade, process), group in data.groupby(
        ["grade", "process"]
    ):
        group = group.sort_values("date")

        plt.plot(
            group["date"],
            group["price_usd_lb"],
            label=f"{grade} - {process}",
        )

    plt.xlabel("Date")
    plt.ylabel("Coffee Price (USD/lb)")
    plt.title(
        "Historical Sidama Coffee Export Prices"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path / "price_history.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(
    data,
    horizon=5,
    training=True,
):
    """Create lag, rolling, seasonal, and forecasting features."""

    data = data.copy()

    # Keep every grade/process series separate
    groups = data.groupby(
        ["grade", "process"]
    )["price_usd_lb"]

    # --------------------------------------------------------
    # Current price
    # --------------------------------------------------------

    # Price available at the forecasting time
    data["current_price"] = data["price_usd_lb"]

    # --------------------------------------------------------
    # Historical lag features
    # --------------------------------------------------------

    for lag in [1, 2, 3, 5, 10]:

        data[f"price_lag_{lag}"] = groups.shift(lag)

    # --------------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------------

    data["price_average_5"] = groups.transform(
        lambda values:
        values.shift(1)
        .rolling(
            5,
            min_periods=2,
        )
        .mean()
    )

    data["price_volatility_5"] = groups.transform(
        lambda values:
        values.shift(1)
        .rolling(
            5,
            min_periods=2,
        )
        .std()
    )

    # --------------------------------------------------------
    # Seasonal features
    # --------------------------------------------------------

    data["month"] = data["date"].dt.month

    data["week"] = (
        data["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # Cyclical month encoding
    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    # Cyclical week encoding
    data["week_sin"] = np.sin(
        2 * np.pi * data["week"] / 52
    )

    data["week_cos"] = np.cos(
        2 * np.pi * data["week"] / 52
    )

    # --------------------------------------------------------
    # Forecasting target
    # --------------------------------------------------------

    if training:

        # Predict the price after the selected
        # number of future observations.
        data["target_price"] = groups.shift(
            -horizon
        )

    return data


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_columns(data):
    """Return numerical and categorical model features."""

    numeric_features = [
        "current_price",
        "price_lag_1",
        "price_lag_2",
        "price_lag_3",
        "price_lag_5",
        "price_lag_10",
        "price_average_5",
        "price_volatility_5",
        "month_sin",
        "month_cos",
        "week_sin",
        "week_cos",
    ]

    # Add optional variables only if they exist
    for column in OPTIONAL_NUMERIC_COLUMNS:

        if column in data.columns:
            numeric_features.append(column)

    categorical_features = [
        "grade",
        "process",
    ]

    return (
        numeric_features,
        categorical_features,
    )


# ============================================================
# MODEL
# ============================================================

def build_model(
    numeric_features,
    categorical_features,
):
    """Build preprocessing and Random Forest pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "numbers",
                Pipeline(
                    steps=[
                        (
                            "fill_missing",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                    ]
                ),
                numeric_features,
            ),

            (
                "categories",
                Pipeline(
                    steps=[
                        (
                            "fill_missing",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),

                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "prepare_data",
                preprocessor,
            ),

            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def calculate_metrics(
    actual,
    predictions,
):
    """Calculate regression evaluation metrics."""

    return {
        "mae": round(
            float(
                mean_absolute_error(
                    actual,
                    predictions,
                )
            ),
            4,
        ),

        "rmse": round(
            float(
                mean_squared_error(
                    actual,
                    predictions,
                )
                ** 0.5
            ),
            4,
        ),

        "r2": round(
            float(
                r2_score(
                    actual,
                    predictions,
                )
            ),
            4,
        ),
    }


# ============================================================
# TRAINING
# ============================================================

def train_model(
    input_file,
    output_folder,
    horizon,
):
    """Train and evaluate the forecasting model."""

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_data(input_file)

    print("\nDATASET INFORMATION")
    print("=" * 60)
    print(f"Total Sidama rows: {len(data)}")
    print(
        f"Date range: "
        f"{data['date'].min().date()} "
        f"to "
        f"{data['date'].max().date()}"
    )

    print(
        f"Grades: "
        f"{', '.join(sorted(data['grade'].unique()))}"
    )

    print(
        f"Processes: "
        f"{', '.join(sorted(data['process'].unique()))}"
    )

    # Historical price visualization
    create_price_history_plot(
        data,
        output_folder,
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    data = create_features(
        data,
        horizon=horizon,
        training=True,
    )

    # Remove rows where required forecasting
    # features or future target are unavailable
    data = data.dropna(
        subset=[
            "current_price",
            "target_price",
        ]
    )

    # IMPORTANT:
    # Sort chronologically for time-based evaluation
    data = data.sort_values(
        "date"
    ).reset_index(drop=True)

    if len(data) < 60:
        raise ValueError(
            "You need at least 60 usable price rows. "
            "More historical data will improve the model."
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    (
        numeric_features,
        categorical_features,
    ) = get_feature_columns(data)

    feature_columns = (
        numeric_features
        + categorical_features
    )

    # --------------------------------------------------------
    # Chronological train/test split
    # --------------------------------------------------------

    split_index = int(
        len(data) * 0.8
    )

    train_data = data.iloc[
        :split_index
    ].copy()

    test_data = data.iloc[
        split_index:
    ].copy()

    print("\nTRAIN / TEST SPLIT")
    print("=" * 60)
    print(
        f"Training rows: {len(train_data)}"
    )
    print(
        f"Testing rows: {len(test_data)}"
    )
    print(
        f"Test period: "
        f"{test_data['date'].min().date()} "
        f"to "
        f"{test_data['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Build and train model
    # --------------------------------------------------------

    pipeline = build_model(
        numeric_features,
        categorical_features,
    )

    pipeline.fit(
        train_data[feature_columns],
        train_data["target_price"],
    )

    # --------------------------------------------------------
    # Random Forest predictions
    # --------------------------------------------------------

    predictions = pipeline.predict(
        test_data[feature_columns]
    )

    actual_prices = (
        test_data["target_price"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Naive baseline
    # --------------------------------------------------------
    #
    # For a 5-observation-ahead forecast,
    # persistence predicts that the future price
    # will equal the current price.
    # --------------------------------------------------------

    naive_predictions = (
        test_data["current_price"]
        .to_numpy()
    )

    model_metrics = calculate_metrics(
        actual_prices,
        predictions,
    )

    naive_metrics = calculate_metrics(
        actual_prices,
        naive_predictions,
    )

    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    metrics = {

        "forecast_horizon_observations": horizon,

        "training_rows": int(
            len(train_data)
        ),

        "test_rows": int(
            len(test_data)
        ),

        "model_mae_usd_lb":
            model_metrics["mae"],

        "model_rmse_usd_lb":
            model_metrics["rmse"],

        "model_r2":
            model_metrics["r2"],

        "naive_mae_usd_lb":
            naive_metrics["mae"],

        "naive_rmse_usd_lb":
            naive_metrics["rmse"],

        "naive_r2":
            naive_metrics["r2"],

        "test_start_date":
            str(
                test_data["date"]
                .min()
                .date()
            ),

        "test_end_date":
            str(
                test_data["date"]
                .max()
                .date()
            ),
    }

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path = Path(
        output_folder
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    joblib.dump(
        {
            "model": pipeline,
            "features": feature_columns,
            "horizon": horizon,
        },
        output_path
        / "sidama_price_model.joblib",
    )

    # --------------------------------------------------------
    # Backtest predictions
    # --------------------------------------------------------

    backtest = pd.DataFrame(
        {
            "date":
                test_data["date"],

            "grade":
                test_data["grade"],

            "process":
                test_data["process"],

            "actual_price_usd_lb":
                actual_prices,

            "predicted_price_usd_lb":
                predictions,

            "naive_price_usd_lb":
                naive_predictions,
        }
    )

    backtest.to_csv(
        output_path
        / "backtest_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "Model": [
                "Naive Forecast",
                "Random Forest",
            ],

            "MAE": [
                naive_metrics["mae"],
                model_metrics["mae"],
            ],

            "RMSE": [
                naive_metrics["rmse"],
                model_metrics["rmse"],
            ],

            "R2": [
                naive_metrics["r2"],
                model_metrics["r2"],
            ],
        }
    )

    comparison.to_csv(
        output_path
        / "model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Actual vs predicted plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        backtest["date"],
        backtest[
            "actual_price_usd_lb"
        ],
        label="Actual Price",
    )

    plt.plot(
        backtest["date"],
        backtest[
            "predicted_price_usd_lb"
        ],
        label="Random Forest Forecast",
    )

    plt.plot(
        backtest["date"],
        backtest[
            "naive_price_usd_lb"
        ],
        label="Naive Forecast",
        linestyle="--",
    )

    plt.xlabel("Date")
    plt.ylabel(
        "Coffee Price (USD/lb)"
    )

    plt.title(
        "Sidama Coffee Price: "
        "Actual vs Forecast"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path
        / "actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Residual analysis
    # --------------------------------------------------------

    residuals = (
        actual_prices
        - predictions
    )

    residual_data = pd.DataFrame(
        {
            "date":
                test_data["date"],

            "actual_price_usd_lb":
                actual_prices,

            "predicted_price_usd_lb":
                predictions,

            "residual":
                residuals,
        }
    )

    residual_data.to_csv(
        output_path
        / "residual_analysis.csv",
        index=False,
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.scatter(
        test_data["date"],
        residuals,
    )

    plt.xlabel("Date")
    plt.ylabel(
        "Prediction Error (USD/lb)"
    )

    plt.title(
        "Random Forest Forecast Errors"
    )

    plt.tight_layout()

    plt.savefig(
        output_path
        / "forecast_errors.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    with open(
        output_path / "metrics.json",
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FORECASTING RESULTS")
    print("=" * 60)

    print("\nRandom Forest:")
    print(
        f"MAE:  "
        f"{model_metrics['mae']}"
    )
    print(
        f"RMSE: "
        f"{model_metrics['rmse']}"
    )
    print(
        f"R²:   "
        f"{model_metrics['r2']}"
    )

    print("\nNaive Baseline:")
    print(
        f"MAE:  "
        f"{naive_metrics['mae']}"
    )
    print(
        f"RMSE: "
        f"{naive_metrics['rmse']}"
    )
    print(
        f"R²:   "
        f"{naive_metrics['r2']}"
    )

    print("\n")
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(
        f"\nOutput directory: "
        f"{output_path.resolve()}"
    )

    print(
        "\nTraining complete."
    )


# ============================================================
# FORECASTING
# ============================================================

def make_forecast(
    input_file,
    model_file,
    output_file,
):
    """Generate forecasts using a saved model."""

    saved = joblib.load(
        model_file
    )

    data = load_data(
        input_file
    )

    data = create_features(
        data,
        horizon=saved["horizon"],
        training=False,
    )

    # Select newest observation
    # for every grade/process series
    latest = (
        data
        .dropna(
            subset=["current_price"]
        )
        .groupby(
            ["grade", "process"]
        )
        .tail(1)
        .copy()
    )

    if latest.empty:
        raise ValueError(
            "Not enough historical rows "
            "to make a forecast."
        )

    # Generate forecasts
    latest[
        "forecast_price_usd_lb"
    ] = saved["model"].predict(
        latest[saved["features"]]
    )

    latest[
        "forecast_horizon_observations"
    ] = saved["horizon"]

    output_columns = [
        "date",
        "origin",
        "grade",
        "process",
        "price_usd_lb",
        "forecast_price_usd_lb",
        "forecast_horizon_observations",
    ]

    latest[
        output_columns
    ].to_csv(
        output_file,
        index=False,
    )

    print("\n")
    print("=" * 60)
    print("FORECAST COMPLETE")
    print("=" * 60)

    print(
        latest[
            output_columns
        ].to_string(
            index=False
        )
    )

    print(
        f"\nForecast saved to: "
        f"{Path(output_file).resolve()}"
    )


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Sidama Ethiopian coffee "
            "price forecasting model"
        )
    )

    commands = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    # --------------------------------------------------------
    # TRAIN COMMAND
    # --------------------------------------------------------

    train_command = commands.add_parser(
        "train"
    )

    train_command.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file.",
    )

    train_command.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for model outputs.",
    )

    train_command.add_argument(
        "--horizon",
        type=int,
        default=5,
        help=(
            "Number of future observations "
            "to forecast ahead."
        ),
    )

    # --------------------------------------------------------
    # FORECAST COMMAND
    # --------------------------------------------------------

    forecast_command = commands.add_parser(
        "forecast"
    )

    forecast_command.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file.",
    )

    forecast_command.add_argument(
        "--model",
        required=True,
        help="Path to the saved model.",
    )

    forecast_command.add_argument(
        "--output",
        default="sidama_forecast.csv",
        help="Output CSV for forecasts.",
    )

    args = parser.parse_args()

    if args.command == "train":

        train_model(
            input_file=args.input,
            output_folder=args.output_dir,
            horizon=args.horizon,
        )

    elif args.command == "forecast":

        make_forecast(
            input_file=args.input,
            model_file=args.model,
            output_file=args.output,
        )


if __name__ == "__main__":
    main()
