# %%
import os

# %%
def save_model_comparison(comparison_df):
    os.makedirs("results", exist_ok=True)

    comparison_df.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    print(
        "\nSaved: results/model_comparison.csv"
    )

# %%
def save_feature_importance(
        importance_df,
        crop_name
):
    os.makedirs("results", exist_ok=True)

    importance_df.to_csv(
        f"results/feature_importance_{crop_name.lower()}.csv",
        index=False
    )

    print(
        f"Saved: feature_importance_{crop_name.lower()}.csv"
    )