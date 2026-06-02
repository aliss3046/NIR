# %%
import pandas as pd

# %%
def show_summary_statistics(df):
    """
    Выводит описательную статистику по итоговому датасету.
    """

    print("\nSUMMARY STATISTICS:")
    numeric_df = df.drop(columns=["year"])

    print(numeric_df.describe())

# %%
def show_correlation_matrix(df):
    """
    Выводит корреляционную матрицу.
    """

    print("\nCORRELATION MATRIX:")
    print(df.corr(numeric_only=True))