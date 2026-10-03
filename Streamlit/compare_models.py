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
    "Base": (0.55, 0.35, 0.10),
    "Equal weights": (1 / 3, 1 / 3, 1 / 3),
    "No funding": (0.5, 0.5, 0.0),
}
MODELS = ["Polynomial ridge", "Linear ridge", "Random Forest", "XGBoost", "Naive"]
DEGREES = [1, 2, 3, 4, 5]
ALPHA = 0.1
FIRST_YEAR = 2017
TEST_YEAR = 2024
SEED = 42
EXAMPLE_SUBFIELD = "quantum cryptography"
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


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
    if model_name == "Naive":
        # no-change forecast: next year equals the last training year
        return float(y_train[-1])
    if model_name == "Random Forest":
        return float(RandomForestRegressor(random_state=SEED).fit(X_train, y_train).predict(X_test)[0])
    return float(XGBRegressor(random_state=SEED).fit(X_train, y_train).predict(X_test)[0])


def rmse(e):
    return float(np.sqrt((e ** 2).mean()))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    combined, label_counts = prepare_data()
    label_counts = label_counts[~label_counts["Label"].isin(["error", "invalid_label"])]
    financial_by_year = combined.groupby("Year")["Financial"].first()

    rows = []
    for _, lc in label_counts.iterrows():
        label = lc["Label"]
        degree = determine_degree(int(lc["Total_Count"]))
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
            row = {"label": label, "scheme": scheme}
            for m in MODELS:
                row[m] = abs(predict(m, degree, X_train, y_train, X_test) - actual)
            for d in DEGREES:
                row[f"Degree {d}"] = abs(ridge_poly(d, X_train, y_train, X_test) - actual)
            rows.append(row)
    errors = pd.DataFrame(rows)

    # Table tab:model_comparison and the per-subfield win counts quoted in the text
    model_rows = []
    for scheme, g in errors.groupby("scheme", sort=False):
        best = g[MODELS].idxmin(axis=1)
        for m in MODELS:
            model_rows.append({"scheme": scheme, "model": m, "rmse": rmse(g[m]), "best_in_n_subfields": int((best == m).sum())})
    pd.DataFrame(model_rows).to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False, float_format="%.4f")

    # Fixed-degree RMSEs quoted in the text (base weights)
    base = errors[errors["scheme"] == "Base"]
    degree_rows = [{"degree": d, "rmse": rmse(base[f"Degree {d}"])} for d in DEGREES]
    degree_rows.append({"degree": "rule", "rmse": rmse(base["Polynomial ridge"])})
    pd.DataFrame(degree_rows).to_csv(os.path.join(out_dir, "fixed_degree.csv"), index=False, float_format="%.4f")

    # Appendix table tab:quantum_crypto_models and the example quoted in the text (base weights)
    example = base[base["label"] == EXAMPLE_SUBFIELD][MODELS + ["Degree 2", "Degree 5"]].T.reset_index()
    example.columns = ["model", "abs_error"]
    example.to_csv(os.path.join(out_dir, "quantum_cryptography.csv"), index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
