# %%
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# %%
def run_feature_selection_experiment(df):

    crops = {
        "Wheat": "wheat_mean",
        "Maize": "maize_mean",
        "Barley": "barley_mean",
        "Sorghum": "sorghum_mean",
        "Rice": "rice_mean"
    }

    selected_features = [
        "oil_mean",
        "wheat_mean",
        "maize_mean",
        "barley_mean",
        "sorghum_mean",
        "rice_mean"
    ]

    results = []

    print("\nFEATURE SELECTION EXPERIMENT")

    for crop_name, target in crops.items():

        features = [
            col for col in selected_features
            if col != target
        ]

        model_df = df[
            features + [target]
        ].dropna()

        X = model_df[features]
        y = model_df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            shuffle=False
        )

        model = LinearRegression()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = mean_squared_error(
            y_test,
            y_pred
        ) ** 0.5

        r2 = r2_score(
            y_test,
            y_pred
        )

        results.append({
            "crop": crop_name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 2)
        })

    results_df = pd.DataFrame(results)

    print(results_df)

    results_df.to_csv(
        "results/feature_selection_experiment.csv",
        index=False
    )

    return results_df