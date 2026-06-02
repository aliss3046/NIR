# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# %%
def create_future_features(df, start_year=2025, end_year=2030):
    """
    Создает будущие значения признаков на основе линейного тренда.
    """

    future_years = pd.DataFrame({
        "year": list(range(start_year, end_year + 1))
    })

    future_df = future_years.copy()

    numeric_columns = [
        col for col in df.columns
        if col != "year"
    ]

    for col in numeric_columns:
        temp = df[["year", col]].dropna()

        if len(temp) < 2:
            continue

        model = LinearRegression()

        X = temp[["year"]]
        y = temp[col]

        model.fit(X, y)

        future_df[col] = model.predict(
            future_df[["year"]]
        )

    return future_df

# %%
def forecast_prices_until_2030(df):
    """
    Прогнозирует цены культур до 2030 года через Random Forest.
    """

    crops = {
        "Wheat": "wheat_mean",
        "Maize": "maize_mean",
        "Barley": "barley_mean",
        "Sorghum": "sorghum_mean",
        "Rice": "rice_mean"
    }

    future_df = create_future_features(df)

    forecast_table = pd.DataFrame({
        "year": future_df["year"]
    })

    for crop_name, target_column in crops.items():

        if target_column not in df.columns:
            continue

        feature_columns = [
            col for col in df.columns
            if col not in ["year", target_column]
        ]

        model_df = df[
            feature_columns + [target_column]
        ].dropna()

        X_train = model_df[feature_columns]
        y_train = model_df[target_column]

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42
        )

        model.fit(X_train, y_train)

        X_future = future_df[feature_columns]

        forecast = model.predict(X_future)

        forecast_table[crop_name] = forecast

    print("\nFORECAST PRICES UNTIL 2030:")
    print(forecast_table)

    forecast_table.to_csv(
        "data/processed/forecast_2025_2030.csv",
        index=False
    )

    return forecast_table

# %%
def plot_future_price_forecast(forecast_table):
    """
    График прогноза цен до 2030 года.
    """

    plt.figure(figsize=(12, 6))

    for col in forecast_table.columns:
        if col != "year":
            plt.plot(
                forecast_table["year"],
                forecast_table[col],
                marker="o",
                label=col
            )

    plt.title("Crop Price Forecast 2025-2030")
    plt.xlabel("Year")
    plt.ylabel("Forecast Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()