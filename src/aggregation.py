# %%
import pandas as pd

# %%
def aggregate_prices_by_year(df):
    """
    Создает годовые признаки цен:
    mean, max, min для всех культур.
    """

    yearly_prices = (
        df.groupby("year")
        .agg({
            "Oil": ["mean", "max", "min"],
            "Wheat": ["mean", "max", "min"],
            "Maize": ["mean", "max", "min"],
            "Barley": ["mean", "max", "min"],
            "Sorghum": ["mean", "max", "min"],
            "Rice": ["mean", "max", "min"]
        })
        .reset_index()
    )

    yearly_prices.columns = [
        "year",

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
        "rice_min"
    ]

    return yearly_prices