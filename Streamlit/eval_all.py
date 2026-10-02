import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from Backend import prepare_data, determine_degree

SCHEMES = {
    "0.55/0.35/0.10": (0.55, 0.35, 0.10),
    "1/3 each": (1 / 3, 1 / 3, 1 / 3),
    "0.5/0.5/0": (0.5, 0.5, 0.0),
}
ALPHA = 0.1


def main():
    combined, label_counts = prepare_data()
    label_counts = label_counts[~label_counts["Label"].isin(["error", "invalid_label"])]

    rows = []
    for _, lc in label_counts.iterrows():
        label = lc["Label"]
        total_count = int(lc["Total_Count"])
        degree = determine_degree(total_count)

        df = combined[(combined["Label"] == label) & (combined["Year"] >= 2017) & (combined["Year"] <= 2024)]
        df = df.sort_values("Year")
        train_df = df[df["Year"] <= 2023]
        val_df = df[df["Year"] == 2024]
        if train_df.empty:
            continue

        for scheme, (w_p, w_r, w_f) in SCHEMES.items():
            score = w_p * df["Count_patents"] + w_r * df["Count_research"] + w_f * df["Financial"]

            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train = poly.fit_transform((train_df["Year"].values - 2017).reshape(-1, 1))
            y_train = score.loc[train_df.index].values

            model = Ridge(alpha=ALPHA)
            model.fit(X_train, y_train)

            if val_df.empty:
                val_abs_error = np.nan
            else:
                X_val = poly.transform((val_df["Year"].values - 2017).reshape(-1, 1))
                y_val = score.loc[val_df.index].values
                val_abs_error = float(np.abs(y_val - model.predict(X_val)).mean())

            pred_2028 = float(model.predict(poly.transform(np.array([[2028 - 2017]])))[0])

            rows.append({
                "label": label,
                "scheme": scheme,
                "degree": degree,
                "total_count": total_count,
                "val_abs_error": val_abs_error,
                "pred_2028": pred_2028,
            })

    pd.DataFrame(rows, columns=["label", "scheme", "degree", "total_count", "val_abs_error", "pred_2028"]).to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
