import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

from Backend import prepare_data, determine_degree

SCHEMES = {
    "0.55/0.35/0.10": (0.55, 0.35, 0.10),
    "1/3 each": (1 / 3, 1 / 3, 1 / 3),
    "0.5/0.5/0": (0.5, 0.5, 0.0),
}
MODELS = ["Polynomial ridge", "Linear ridge", "Random Forest", "XGBoost"]
DEGREES = [1, 2, 3, 4, 5]
ALPHA = 0.1
FIRST_YEAR = 2017
TEST_YEAR = 2024
SEED = 42


def ridge_poly(degree, X_train, y_train, X_test):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    model = Ridge(alpha=ALPHA)
    model.fit(poly.fit_transform(X_train), y_train)
    return float(model.predict(poly.transform(X_test))[0])


def predict(model_name, degree, X_train, y_train, X_test):
    if model_name == "Polynomial ridge":
        return ridge_poly(degree, X_train, y_train, X_test)
    if model_name == "Linear ridge":
        return ridge_poly(1, X_train, y_train, X_test)
    if model_name == "Random Forest":
        return float(RandomForestRegressor(random_state=SEED).fit(X_train, y_train).predict(X_test)[0])
    return float(XGBRegressor(random_state=SEED).fit(X_train, y_train).predict(X_test)[0])


def summarize(errors, names):
    rows = []
    best = errors[names].idxmin(axis=1)
    for name in names:
        e = errors[name]
        rows.append({
            "model": name,
            "n_subfields": len(e),
            "MAE": e.mean(),
            "RMSE": np.sqrt((e ** 2).mean()),
            "MedianAE": e.median(),
            "best_in_n_subfields": int((best == name).sum()),
        })
    return pd.DataFrame(rows)


def to_markdown(df):
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "model_comparison"
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
        df = df.reindex(range(FIRST_YEAR, TEST_YEAR + 1))
        df[["Count_patents", "Count_research"]] = df[["Count_patents", "Count_research"]].fillna(0)
        df["Financial"] = df["Financial"].fillna(financial_by_year)
        years = np.array(df.index, dtype=float)
        X_train = (years[years < TEST_YEAR] - FIRST_YEAR).reshape(-1, 1)
        X_test = np.array([[TEST_YEAR - FIRST_YEAR]], dtype=float)
        for scheme, (w_p, w_r, w_f) in SCHEMES.items():
            score = (w_p * df["Count_patents"] + w_r * df["Count_research"] + w_f * df["Financial"]).values
            y_train, actual = score[:-1], score[-1]
            row = {"label": label, "total_count": total_count, "degree": degree, "scheme": scheme, "actual_2024": actual}
            for m in MODELS:
                row[f"pred_{m}"] = predict(m, degree, X_train, y_train, X_test)
                row[m] = abs(row[f"pred_{m}"] - actual)
            for d in DEGREES:
                row[f"pred_degree_{d}"] = ridge_poly(d, X_train, y_train, X_test)
                row[f"Degree {d}"] = abs(row[f"pred_degree_{d}"] - actual)
            rows.append(row)

    errors = pd.DataFrame(rows)
    errors.to_csv(os.path.join(out_dir, "per_subfield_errors.csv"), index=False)

    model_summary = pd.concat([summarize(g, MODELS).assign(scheme=s) for s, g in errors.groupby("scheme", sort=False)])
    model_summary = model_summary[["scheme"] + [c for c in model_summary.columns if c != "scheme"]]
    model_summary.to_csv(os.path.join(out_dir, "table_models.csv"), index=False)

    degree_names = [f"Degree {d}" for d in DEGREES]
    degree_summary = pd.concat([summarize(g, degree_names).assign(scheme=s) for s, g in errors.groupby("scheme", sort=False)])
    degree_summary = degree_summary[["scheme"] + [c for c in degree_summary.columns if c != "scheme"]]
    degree_summary.to_csv(os.path.join(out_dir, "table_degrees.csv"), index=False)

    base = errors[errors["scheme"] == "0.55/0.35/0.10"][["label", "total_count", "degree", "actual_2024"] + MODELS]
    base.to_csv(os.path.join(out_dir, "table_per_subfield_base.csv"), index=False)

    with open(os.path.join(out_dir, "tables.md"), "w") as f:
        f.write("## Models: train 2017-2023, test 2024, all subfields\n\n" + to_markdown(model_summary) + "\n\n")
        f.write("## Polynomial degree: same setup, one fixed degree for every subfield\n\n" + to_markdown(degree_summary) + "\n\n")
        f.write("## Per-subfield absolute error on 2024, base weights\n\n" + to_markdown(base) + "\n")

    print(to_markdown(model_summary))
    print()
    print(to_markdown(degree_summary))


if __name__ == "__main__":
    main()
