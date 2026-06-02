# %%
import pandas as pd

# %%
def compare_models(linear_metrics, arima_metrics, rf_metrics):

    comparison = pd.concat(
        [linear_metrics, arima_metrics, rf_metrics],
        ignore_index=True
    )

    print("\nFULL MODEL COMPARISON")
    print(comparison)

    print("\nBEST MODEL FOR EACH CROP")

    for crop in comparison["crop"].unique():
        crop_df = comparison[comparison["crop"] == crop]

        best_model = crop_df.loc[crop_df["MAE"].idxmin()]

        print(
            crop,
            "->",
            best_model["model"],
            "(MAE =",
            round(best_model["MAE"], 2),
            ")"
        )
    best_models = []
    best_models.append({
        "crop": crop,
        "best_model": best_model["model"],
        "MAE": best_model["MAE"]
    })
    best_models_df = pd.DataFrame(
        best_models
    )

    best_models_df.to_csv(
        "results/best_models.csv",
        index=False
    )

    return comparison