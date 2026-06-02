# %%
import pandas as pd

# %%
def clean_faostat(df):
    """
    Очистка FAOSTAT: оставляем основные зерновые культуры.
    """

    df = df[["Area", "Item", "Year", "Value"]]

    df.columns = ["country", "item", "year", "production"]

    df = df.dropna()

    df["year"] = df["year"].astype(int)

    df["production"] = pd.to_numeric(
        df["production"],
        errors="coerce"
    )

    df = df[df["production"] > 0]

    df["item"] = df["item"].astype(str).str.strip()

    allowed_items = [
        "Wheat",
        "Maize (corn)",
        "Barley",
        "Rice",
        "Rice, paddy",
        "Sorghum",
        "Rye",
        "Oats",
        "Millet",
        "Buckwheat"
    ]

    df = df[df["item"].isin(allowed_items)]

    df["item"] = df["item"].replace({
        "Maize (corn)": "maize",
        "Wheat": "wheat",
        "Barley": "barley",
        "Rice": "rice",
        "Rice, paddy": "rice",
        "Sorghum": "sorghum",
        "Rye": "rye",
        "Oats": "oats",
        "Millet": "millet",
        "Buckwheat": "buckwheat"
    })

    return df

# %%
def clean_prices(df):
    df = df.copy()

    df.columns = [
        "Date",
        "Oil",
        "Wheat",
        "Maize",
        "Barley",
        "Sorghum",
        "Rice"
    ]

    dates = pd.date_range(
        start="1960-01-01",
        periods=len(df),
        freq="MS"
    )

    df["year"] = dates.year

    price_columns = [
        "Oil",
        "Wheat",
        "Maize",
        "Barley",
        "Sorghum",
        "Rice"
    ]

    for col in price_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        [
            "year",
            "Oil",
            "Wheat",
            "Maize",
            "Barley",
            "Sorghum",
            "Rice"
        ]
    ]

    return df

# %%
def fix_world_bank_dates(df):

    df = df.copy()

    # переименовываем первую колонку
    df = df.rename(columns={df.columns[0]: "Date"})

    # ГОД
    df["year"] = (
        df["Date"]
        .astype(str)
        .str.split("M")
        .str[0]
    )

    # МЕСЯЦ
    df["month"] = (
        df["Date"]
        .astype(str)
        .str.split("M")
        .str[1]
    )

    # перевод в числа
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    return df
