# %%
import pandas as pd
import matplotlib.pyplot as plt
import math 
from sklearn.ensemble import RandomForestRegressor

# %%
def analyze_feature_importance(
    df,
    target_column,
    crop_name
):

    features = []

    for col in df.columns:

        if col == "year":
            continue

        if col == target_column:
            continue

        if col.endswith("_max"):
            continue

        if col.endswith("_min"):
            continue

        features.append(col)

    model_df = df[
        features + [target_column]
    ].dropna()

    X = model_df[features]
    y = model_df[target_column]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    print(f"\nFEATURE IMPORTANCE FOR {crop_name.upper()}")
    print(importance_df)

    return importance_df

# %%
def plot_all_feature_importance(importance_results):
    """
    Рисует Feature Importance для всех культур в одном окне.
    """

    n = len(importance_results)

    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 5 * rows)
    )

    axes = axes.flatten()

    for ax, (crop_name, importance_df) in zip(
        axes,
        importance_results.items()
    ):
        top_features = importance_df.head(10)

        ax.barh(
            top_features["feature"],
            top_features["importance"]
        )

        ax.set_title(f"{crop_name} Feature Importance")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        ax.grid(True, axis="x")

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=3)
    from src.save_plots import save_current_figure

    save_current_figure(
        "forecast_2025_2030.png"
    )
    plt.show()