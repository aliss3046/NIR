# %%
import pandas as pd

# %%
def prepare_production_for_merge(df):
    """
    Преобразует FAOSTAT из длинного формата в широкий:
    year | wheat_production | maize_production | ...
    """

    production_wide = (
        df.pivot_table(
            index="year",
            columns="item",
            values="production",
            aggfunc="sum"
        )
        .reset_index()
    )

    production_wide.columns.name = None

    production_wide = production_wide.rename(columns={
        "wheat": "wheat_production",
        "maize": "maize_production",
        "barley": "barley_production",
        "rice": "rice_production",
        "sorghum": "sorghum_production",
        "rye": "rye_production",
        "oats": "oats_production",
        "millet": "millet_production",
        "buckwheat": "buckwheat_production"
    })

    return production_wide

# %%
def merge_datasets(price_df, production_df):
    """
    Объединяет годовые цены и производство по году.
    """

    production_wide = prepare_production_for_merge(production_df)

    final_df = pd.merge(
        price_df,
        production_wide,
        on="year",
        how="inner"
    )

    return final_df