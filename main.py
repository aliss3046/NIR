# %%
import pandas as pd

from src.data_loader import load_production_data, load_price_data
from src.preprocessing import clean_faostat, clean_prices
from src.aggregation import aggregate_prices_by_year
from src.merge_data import merge_datasets
from src.analysis import show_summary_statistics, show_correlation_matrix
from src.visualization import (
    plot_project_dashboard,
    plot_correlation_heatmap
)
from src.time_series_analysis import moving_average_analysis
from src.forecasting import run_linear_regression_models, plot_all_predictions
from src.arima_forecasting import run_arima_models, plot_arima_results
from src.random_forest_model import (
    run_random_forest_models,
    plot_random_forest_results
)
from src.feature_importance import (
    analyze_feature_importance,
    plot_all_feature_importance
)
from src.model_comparison import compare_models
from src.future_forecast import (
    forecast_prices_until_2030,
    plot_future_price_forecast
)
from src.save_results import (
    save_model_comparison,
    save_feature_importance
)
from src.feature_selection_experiment import (
    run_feature_selection_experiment
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

# %%
def main():
    print("Loading data...")

    production_df = load_production_data()
    price_df = load_price_data()

    print("Cleaning data...")

    production_df = clean_faostat(production_df)
    print("\nFAOSTAT ITEMS AFTER CLEANING:")
    print(production_df["item"].value_counts())
    price_df = clean_prices(price_df)

    print("Aggregating price data...")

    price_df = aggregate_prices_by_year(price_df)

    print("Merging datasets...")

    final_df = merge_datasets(price_df, production_df)

    final_df.to_csv(
        "data/processed/final_dataset.csv",
        index=False
    )

    print("\nFinal dataset saved to data/processed/final_dataset.csv")

    print("\nFINAL DATASET:")
    print(final_df.head())

    show_summary_statistics(final_df)
    show_correlation_matrix(final_df)

    plot_project_dashboard(final_df)
    plot_correlation_heatmap(final_df)
    moving_average_analysis(final_df)
    forecast_results = run_linear_regression_models(final_df)
    forecast_metrics = forecast_results["metrics"]
    plot_all_predictions(forecast_results)

    arima_results, arima_metrics = run_arima_models(final_df)
    plot_arima_results(arima_results)

    rf_results, rf_metrics = run_random_forest_models(
        final_df
    )

    comparison_df = compare_models(
        forecast_metrics,
        arima_metrics,
        rf_metrics
    )

    save_model_comparison(
        comparison_df
    )

    plot_random_forest_results(
        rf_results
    )

    crops_for_importance = {
        "Wheat": "wheat_mean",
        "Maize": "maize_mean",
        "Barley": "barley_mean",
        "Sorghum": "sorghum_mean",
        "Rice": "rice_mean"
    }

    importance_results = {}

    for crop_name, target_column in crops_for_importance.items():
        if target_column in final_df.columns:

            importance_df = analyze_feature_importance(
                final_df,
                target_column,
                crop_name
            )

            save_feature_importance(
                importance_df,
                crop_name
            )

            importance_results[crop_name] = importance_df

    plot_all_feature_importance(importance_results)
    
    future_forecast = forecast_prices_until_2030(final_df)

    plot_future_price_forecast(future_forecast)

    run_feature_selection_experiment(
        final_df
    )

# %%
if __name__ == "__main__":
    main()