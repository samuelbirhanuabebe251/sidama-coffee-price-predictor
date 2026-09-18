import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REQUIRED_COLUMNS = ["date", "price_usd_lb", "origin", "grade", "process"]

OPTIONAL_NUMERIC_COLUMNS = [
    "usd_etb",
    "arabica_cents_lb",
    "rainfall_mm",
    "temperature_c",
    "export_volume_kg",
]


def load_data(csv_file):
    data = pd.read_csv(csv_file)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "Your CSV is missing these columns: " + ", ".join(missing)
        )

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price_usd_lb"] = pd.to_numeric(
        data["price_usd_lb"], errors="coerce"
    )

    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["origin"] = data["origin"].astype(str).str.strip().str.lower()
    data["grade"] = data["grade"].astype(str).str.strip().str.upper()
    data["process"] = data["process"].astype(str).str.strip().str.lower()

    data = data.dropna(subset=["date", "price_usd_lb"])
    data = data[data["origin"] == "sidama"].copy()

    if data.empty:
        raise ValueError(
            "No Sidama rows found. Use 'Sidama' in the origin column."
        )

    return data.sort_values(["grade", "process", "date"]).reset_index(drop=True)


def create_features(data, horizon=5, training=True):
    data = data.copy()

    # Keep each coffee grade/process series separate.
    groups = data.groupby(["grade", "process"])["price_usd_lb"]

    # Historical price features. shift(1) prevents using future prices.
    for lag in [1, 2, 3, 5, 10]:
        data[f"price_lag_{lag}"] = groups.shift(lag)

    data["price_average_5"] = groups.transform(
        lambda values: values.shift(1).rolling(5, min_periods=2).mean()
    )

    data["price_volatility_5"] = groups.transform(
        lambda values: values.shift(1).rolling(5, min_periods=2).std()
    )

    # Seasonal features.
    data["month"] = data["date"].dt.month
    data["week"] = data["date"].dt.isocalendar().week.astype(int)

    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["week_sin"] = np.sin(2 * np.pi * data["week"] / 52)
    data["week_cos"] = np.cos(2 * np.pi * data["week"] / 52)

    if training:
        # Predict the price after the selected number of future observations.
        data["target_price"] = groups.shift(-horizon)

    return data


def get_feature_columns(data):
    numeric_features = [
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

    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column in data.columns:
            numeric_features.append(column)

    categorical_features = ["grade", "process"]

    return numeric_features, categorical_features


def build_model(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numbers",
                Pipeline(
                    steps=[
                        ("fill_missing", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "categories",
                Pipeline(
                    steps=[
                        ("fill_missing", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
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
            ("prepare_data", preprocessor),
            ("model", model),
        ]
    )


def train_model(input_file, output_folder, horizon):
    data = load_data(input_file)
    data = create_features(data, horizon=horizon, training=True)

    data = data.dropna(
        subset=["price_lag_1", "target_price"]
    ).sort_values("date")

    if len(data) < 60:
        raise ValueError(
            "You need at least 60 usable price rows. More history will improve the model."
        )

    numeric_features, categorical_features = get_feature_columns(data)
    feature_columns = numeric_features + categorical_features

    # The newest 20% of records are used as a realistic test.
    split_index = int(len(data) * 0.8)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    pipeline = build_model(numeric_features, categorical_features)

    pipeline.fit(
        train_data[feature_columns],
        train_data["target_price"],
    )

    predictions = pipeline.predict(test_data[feature_columns])
    actual_prices = test_data["target_price"].to_numpy()

    # Baseline: predict that future price equals today's price.
    naive_predictions = test_data["price_lag_1"].to_numpy()

    metrics = {
        "forecast_horizon_observations": horizon,
        "training_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "model_mae_usd_lb": round(
            float(mean_absolute_error(actual_prices, predictions)), 4
        ),
        "model_rmse_usd_lb": round(
            float(mean_squared_error(actual_prices, predictions) ** 0.5), 4
        ),
        "naive_mae_usd_lb": round(
            float(mean_absolute_error(actual_prices, naive_predictions)), 4
        ),
        "test_start_date": str(test_data["date"].min().date()),
        "test_end_date": str(test_data["date"].max().date()),
    }

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": pipeline,
            "features": feature_columns,
            "horizon": horizon,
        },
        output_path / "sidama_price_model.joblib",
    )

    backtest = pd.DataFrame(
        {
            "date": test_data["date"],
            "grade": test_data["grade"],
            "process": test_data["process"],
            "actual_price_usd_lb": actual_prices,
            "predicted_price_usd_lb": predictions,
            "naive_price_usd_lb": naive_predictions,
        }
    )

    backtest.to_csv(
        output_path / "backtest_predictions.csv",
        index=False,
    )

    with open(output_path / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print("\nTRAINING COMPLETE\n")
    print(json.dumps(metrics, indent=2))
    print(f"\nFiles saved in: {output_path.resolve()}")


def make_forecast(input_file, model_file, output_file):
    saved = joblib.load(model_file)

    data = load_data(input_file)
    data = create_features(
        data,
        horizon=saved["horizon"],
        training=False,
    )

    # Forecast the newest row for each Sidama grade/process series.
    latest = (
        data.dropna(subset=["price_lag_1"])
        .groupby(["grade", "process"])
        .tail(1)
        .copy()
    )

    if latest.empty:
        raise ValueError("Not enough historical rows to make a forecast.")

    latest["forecast_price_usd_lb"] = saved["model"].predict(
        latest[saved["features"]]
    )

    latest["forecast_horizon_observations"] = saved["horizon"]

    output_columns = [
        "date",
        "origin",
        "grade",
        "process",
        "price_usd_lb",
        "forecast_price_usd_lb",
        "forecast_horizon_observations",
    ]

    latest[output_columns].to_csv(output_file, index=False)

    print("\nFORECAST COMPLETE\n")
    print(latest[output_columns].to_string(index=False))
    print(f"\nForecast saved to: {Path(output_file).resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Sidama Ethiopian coffee price prediction model"
    )

    commands = parser.add_subparsers(dest="command", required=True)

    train_command = commands.add_parser("train")
    train_command.add_argument("--input", required=True)
    train_command.add_argument("--output-dir", default="artifacts")
    train_command.add_argument("--horizon", type=int, default=5)

    forecast_command = commands.add_parser("forecast")
    forecast_command.add_argument("--input", required=True)
    forecast_command.add_argument("--model", required=True)
    forecast_command.add_argument("--output", default="sidama_forecast.csv")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.input, args.output_dir, args.horizon)

    if args.command == "forecast":
        make_forecast(args.input, args.model, args.output)


if __name__ == "__main__":
    main()







    