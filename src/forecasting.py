# %%
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %%
def train_price_model(df, target_column, feature_columns, crop_name):
    """
    Обучает модель линейной регрессии для выбранной культуры.
    """

    model_df = df[
        feature_columns + [target_column, "year"]
    ].copy()

    model_df = model_df.dropna()

    X = model_df[feature_columns]
    y = model_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=False
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nLINEAR REGRESSION RESULTS FOR {crop_name.upper()}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    results = pd.DataFrame({
        "year": df.loc[y_test.index, "year"],
        "actual_price": y_test,
        "predicted_price": y_pred
    })

    print("\nPREDICTIONS:")
    print(results)

    metrics = {
        "crop": crop_name,
        "model": "Linear Regression",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return model, results, metrics
# %%
def run_linear_regression_models(df):
    """
    Запускает линейную регрессию для всех культур,
    по которым есть цены.
    """

    crops = {
        "Wheat": "wheat_mean",
        "Maize": "maize_mean",
        "Barley": "barley_mean",
        "Sorghum": "sorghum_mean",
        "Rice": "rice_mean"
    }

    all_price_features = [
        "oil_mean",
        "oil_max",
        "oil_min",
        "wheat_mean",
        "wheat_max",
        "wheat_min",
        "maize_mean",
        "maize_max",
        "maize_min",
        "barley_mean",
        "barley_max",
        "barley_min",
        "sorghum_mean",
        "sorghum_max",
        "sorghum_min",
        "rice_mean",
        "rice_max",
        "rice_min",
        "wheat_production",
        "maize_production",
        "barley_production",
        "sorghum_production",
        "rice_production"
    ]

    results = {}
    metrics_list = []

    for crop_name, target_column in crops.items():

        feature_columns = [
            col for col in all_price_features
            if col != target_column
        ]

        feature_columns = [
            col for col in feature_columns
            if col in df.columns
        ]

        model, crop_results, crop_metrics = train_price_model(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns,
            crop_name=crop_name
        )

        results[f"{crop_name.lower()}_model"] = model
        results[f"{crop_name.lower()}_results"] = crop_results

        metrics_list.append(crop_metrics)

    metrics_df = pd.DataFrame(metrics_list)

    print("\nMODEL COMPARISON:")
    print(metrics_df)

    results["metrics"] = metrics_df

    return results
# %%
def plot_predictions(results_df, crop_name):

    plt.figure(figsize=(10, 5))

    plt.plot(
        results_df["year"],
        results_df["actual_price"],
        marker="o",
        label="Actual"
    )

    plt.plot(
        results_df["year"],
        results_df["predicted_price"],
        marker="o",
        label="Predicted"
    )

    plt.title(f"{crop_name} Price Forecast")

    plt.xlabel("Year")
    plt.ylabel("Price")

    plt.legend()
    plt.grid(True)
    from src.save_plots import save_current_figure
    save_current_figure(
        "linear_regression_forecasts.png"
    )

    plt.show()
# %%
import math


def plot_all_predictions(forecast_results):
    """
    Строит прогнозы всех культур в одном окне.
    """

    result_items = {
        key: value
        for key, value in forecast_results.items()
        if key.endswith("_results")
    }

    n = len(result_items)

    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(14, 5 * rows)
    )

    axes = axes.flatten()

    for ax, (key, results_df) in zip(axes, result_items.items()):
        crop_name = key.replace("_results", "").capitalize()

        ax.plot(
            results_df["year"],
            results_df["actual_price"],
            marker="o",
            label="Actual"
        )

        ax.plot(
            results_df["year"],
            results_df["predicted_price"],
            marker="o",
            label="Predicted"
        )

        ax.set_title(f"{crop_name} Price Forecast")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend()

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()