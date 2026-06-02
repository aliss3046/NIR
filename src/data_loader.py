# %%
import pandas as pd

# %%
def load_production_data():
    """
    Загрузка данных FAOSTAT
    """

    file_path = "data/raw/wheat_production.csv"

    df = pd.read_csv(file_path)

    return df

# %%
def load_price_data():
    file_path = "data/raw/commodity_prices.xlsx"

    df = pd.read_excel(file_path)

    # print(df.columns)
    # print(df.head())

    return df