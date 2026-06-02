# %%
import math
import matplotlib.pyplot as plt


# %%
def moving_average_analysis(df):
    """
    Скользящие средние для всех культур в одном аккуратном окне.
    """

    crop_columns = [
        "wheat_mean",
        "maize_mean",
        "barley_mean",
        "sorghum_mean",
        "rice_mean"
    ]

    available_crops = [
        col for col in crop_columns
        if col in df.columns
    ]

    n = len(available_crops)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 5 * rows)
    )

    axes = axes.flatten()

    for ax, col in zip(axes, available_crops):
        temp = df[["year", col]].copy()

        ma_col = col.replace("_mean", "_ma3")
        temp[ma_col] = temp[col].rolling(window=3).mean()

        crop_name = col.replace("_mean", "").capitalize()

        ax.plot(
            temp["year"],
            temp[col],
            marker="o",
            linewidth=2,
            label="Actual"
        )

        ax.plot(
            temp["year"],
            temp[ma_col],
            linewidth=3,
            label="MA(3)"
        )

        ax.set_title(f"{crop_name} Price Moving Average")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend()

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=3)
    from src.save_plots import save_current_figure
    save_current_figure(
        "moving_average.png"
    )
    plt.show()