# %%
import matplotlib.pyplot as plt

# %%
def plot_project_dashboard(df):
    """
    Все основные графики в одном окне.
    """

    price_columns = [
        col for col in df.columns
        if col.endswith("_mean") and col != "oil_mean"
    ]

    production_columns = [
        col for col in df.columns
        if col.endswith("_production")
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 10)
    )

    # 1. Цены культур
    for col in price_columns:
        axes[0].plot(
            df["year"],
            df[col],
            marker="o",
            label=col.replace("_mean", "").capitalize()
        )

    axes[0].set_title("Average Crop Prices by Year")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Price")
    axes[0].grid(True)
    axes[0].legend()

    # 2. Производство культур
    for col in production_columns:
        axes[1].plot(
            df["year"],
            df[col],
            marker="o",
            label=col.replace("_production", "").capitalize()
        )

    axes[1].set_title("Crop Production by Year")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Production")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# %%
def plot_correlation_heatmap(df):
    """
    Heatmap корреляционной матрицы.
    """

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))

    heatmap = plt.imshow(corr)

    plt.colorbar(heatmap)

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()
    from src.save_plots import save_current_figure
    save_current_figure(
        "correlation_heatmap.png"
    )
    plt.show()