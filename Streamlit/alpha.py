import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from Backend import prepare_data, determine_degree

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9, 3.0, 4.5]
EXAMPLE_SUBFIELD = "quantum error correction"
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    combined, label_counts = prepare_data()
    label_counts = label_counts[~label_counts["Label"].isin(["error", "invalid_label"])]
    financial_by_year = combined.groupby("Year")["Financial"].first()

    X = (np.arange(2017, 2025) - 2017).reshape(-1, 1).astype(float)
    series = {}
    for _, lc in label_counts.iterrows():
        df = combined[combined["Label"] == lc["Label"]].set_index("Year")[["Count_patents", "Count_research", "Financial"]]
        df = df.reindex(range(2017, 2025))
        df[["Count_patents", "Count_research"]] = df[["Count_patents", "Count_research"]].fillna(0)
        df["Financial"] = df["Financial"].fillna(financial_by_year)
        y = (0.55 * df["Count_patents"] + 0.35 * df["Count_research"] + 0.10 * df["Financial"]).values
        series[lc["Label"]] = (determine_degree(int(lc["Total_Count"])), y)

    # Appendix table tab:alpha-rmse (base weights, train 2017-2023, validate 2024)
    rows = []
    for alpha in ALPHAS:
        errors = {}
        for label, (degree, y) in series.items():
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            model = Ridge(alpha=alpha).fit(poly.fit_transform(X[:-1]), y[:-1])
            errors[label] = float(model.predict(poly.transform(X[-1:]))[0] - y[-1])
        errors = pd.Series(errors)
        rows.append({"alpha": alpha,
                     "rmse_quantum_error_correction": abs(errors[EXAMPLE_SUBFIELD]),
                     "rmse_all_subfields": float(np.sqrt((errors ** 2).mean()))})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "alpha.csv"), index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
