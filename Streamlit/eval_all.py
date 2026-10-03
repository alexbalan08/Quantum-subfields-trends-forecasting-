import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from Backend import prepare_data, determine_degree

SCHEMES = {
    "Base": (0.55, 0.35, 0.10),
    "Equal weights": (1 / 3, 1 / 3, 1 / 3),
    "No funding": (0.5, 0.5, 0.0),
}
ALPHA = 0.1
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    combined, label_counts = prepare_data()
    label_counts = label_counts[~label_counts["Label"].isin(["error", "invalid_label"])]
    financial_by_year = combined.groupby("Year")["Financial"].first()

    rows = []
    for _, lc in label_counts.iterrows():
        label = lc["Label"]
        total_count = int(lc["Total_Count"])
        degree = determine_degree(total_count)

        df = combined[combined["Label"] == label].set_index("Year")[["Count_patents", "Count_research", "Financial"]]
        df = df.reindex(range(2017, 2025))
        df[["Count_patents", "Count_research"]] = df[["Count_patents", "Count_research"]].fillna(0)
        df["Financial"] = df["Financial"].fillna(financial_by_year)
        df = df.rename_axis("Year").reset_index()
        train_df = df[df["Year"] <= 2023]
        val_df = df[df["Year"] == 2024]

        row = {"subfield": label, "records": total_count, "degree": degree}
        for scheme, (w_p, w_r, w_f) in SCHEMES.items():
            score = w_p * df["Count_patents"] + w_r * df["Count_research"] + w_f * df["Financial"]

            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train = poly.fit_transform((train_df["Year"].values - 2017).reshape(-1, 1))
            model = Ridge(alpha=ALPHA).fit(X_train, score.loc[train_df.index].values)

            X_val = poly.transform((val_df["Year"].values - 2017).reshape(-1, 1))
            row[f"error_2024_{scheme}"] = float(np.abs(score.loc[val_df.index].values - model.predict(X_val)).mean())
            row[f"pred_2028_{scheme}"] = float(model.predict(poly.transform(np.array([[2028 - 2017]])))[0])
        rows.append(row)
    results = pd.DataFrame(rows)

    # Appendix table tab:full_ranking
    for scheme in SCHEMES:
        results[f"rank_{scheme}"] = results[f"pred_2028_{scheme}"].rank(ascending=False, method="min").astype(int)
    ranking = results[["subfield", "records", "degree", "pred_2028_Base"]
                      + [f"rank_{s}" for s in SCHEMES] + ["error_2024_Base"]]
    ranking = ranking.sort_values("pred_2028_Base", ascending=False)
    ranking.to_csv(os.path.join(out_dir, "full_ranking.csv"), index=False, float_format="%.4f")

    # Rank correlation of the 2028 ordering between schemes, quoted in the text
    corr = [{"scheme_a": a, "scheme_b": b,
             "spearman": results[f"pred_2028_{a}"].corr(results[f"pred_2028_{b}"], method="spearman")}
            for a, b in combinations(SCHEMES, 2)]
    pd.DataFrame(corr).to_csv(os.path.join(out_dir, "rank_correlation.csv"), index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
