# %%
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


warnings.filterwarnings("ignore")

# %%
def select_best_arima_order(series):
    """
    Подбирает лучшие параметры ARIMA(p,d,q) по AIC.
    """

    best_aic = np.inf
    best_order = None

    p_values = range(0, 4)
    d_values = range(0, 3)
    q_values = range(0, 4)

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(
                        series,
                        order=(p, d, q)
                    )

                    fitted_model = model.fit()

                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)

                except Exception:
                    continue

    return best_order

# %%
def run_arima_for_series(df, target_column, crop_name):
    """
    ARIMA-прогноз для одной культуры с подбором параметров.
    """

    series_df = (
        df[["year", target_column]]
        .dropna()
        .sort_values("year")
        .copy()
    )

    # делаем временной индекс
    series_df["date"] = pd.to_datetime(
        series_df["year"].astype(str) + "-01-01"
    )

    series_df = series_df.set_index("date")

    series = series_df[target_column]

    train = series.iloc[:-5]
    test = series.iloc[-5:]

    best_order = select_best_arima_order(train)

    if best_order is None:
        best_order = (1, 1, 1)

    model = ARIMA(
        train,
        order=best_order
    )

    fitted_model = model.fit()

    forecast = fitted_model.forecast(
        steps=len(test)
    )

    mae = mean_absolute_error(
        test,
        forecast
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            forecast
        )
    )

    results = pd.DataFrame({
        "year": test.index.year,
        "actual": test.values,
        "forecast": forecast.values
    })

    metrics = {
        "crop": crop_name,
        "model": f"ARIMA{best_order}",
        "MAE": mae,
        "RMSE": rmse
    }

    return results, metrics

# %%
def run_arima_models(df):
    """
    Запускает ARIMA для всех культур.
    """

    crops = {
        "Wheat": "wheat_mean",
        "Maize": "maize_mean",
        "Barley": "barley_mean",
        "Sorghum": "sorghum_mean",
        "Rice": "rice_mean"
    }

    all_results = {}
    metrics_list = []

    for crop_name, column in crops.items():

        if column not in df.columns:
            continue

        results, metrics = run_arima_for_series(
            df,
            column,
            crop_name
        )

        all_results[crop_name] = results
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    print("\nARIMA MODEL COMPARISON:")
    print(metrics_df)

    return all_results, metrics_df

# %%
def plot_arima_results(all_results):
    """
    Графики ARIMA-прогнозов в одном окне.
    """

    n = len(all_results)

    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 5 * rows)
    )

    axes = axes.flatten()

    for ax, (crop_name, results) in zip(
        axes,
        all_results.items()
    ):

        ax.plot(
            results["year"],
            results["actual"],
            marker="o",
            linewidth=2,
            label="Actual"
        )

        ax.plot(
            results["year"],
            results["forecast"],
            marker="o",
            linestyle="--",
            linewidth=2,
            label="ARIMA Forecast"
        )

        ax.set_title(
            f"{crop_name}: Actual vs ARIMA Forecast"
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend()

    for i in range(
        len(all_results),
        len(axes)
    ):
        fig.delaxes(axes[i])

    plt.tight_layout()
    from src.save_plots import save_current_figure
    save_current_figure(
        "arima_forecasts.png"
    )
    plt.show()