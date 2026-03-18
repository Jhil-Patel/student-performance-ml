"""
train_model.py
==============
Trains Linear Regression, Decision Tree, and Random Forest on the
UCI Student Performance dataset. Two experiments:

  A — WITH G1/G2 prior grades (clean dataset, n=359) → R² ≈ 0.975
  B — WITHOUT prior grades (full dataset, n=395)     → R² ≈ 0.374

Usage:
    python src/train_model.py

Outputs → models/
    model_with.pkl, model_without.pkl
    feats_with.pkl, feats_without.pkl
    all_results.json, dataset_stats.json
    scatter_data.json, corr_matrix.json, grade_dist.json
"""

import sys, os, json, pickle, warnings
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from preprocess import load_data, get_features_target
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing   import LabelEncoder

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def run_experiment(label, clean, include_prior):
    df = load_data(clean=clean)
    X, y = get_features_target(df, include_prior_grades=include_prior)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model_defs = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=300, max_depth=12,
                                                    min_samples_leaf=2, random_state=42),
    }

    results, trained = {}, {}
    print(f"\n{'='*58}\n  {label}\n{'='*58}")
    print(f"  {'Model':<22} {'R²':>7} {'CV R²':>8} {'RMSE':>7}")

    for name, m in model_defs.items():
        m.fit(Xtr, ytr)
        p    = m.predict(Xte)
        r2   = float(r2_score(yte, p))
        rmse = float(np.sqrt(mean_squared_error(yte, p)))
        mae  = float(mean_absolute_error(yte, p))
        cv   = float(cross_val_score(m, X, y, cv=5, scoring="r2").mean())
        results[name] = dict(r2=round(r2,4), cv_r2=round(cv,4),
                             rmse=round(rmse,4), mae=round(mae,4))
        trained[name] = m
        print(f"  {name:<22} {r2:>7.3f} {cv:>8.3f} {rmse:>7.3f}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\n  🏆 Best: {best_name} (R²={results[best_name]['r2']})")

    rf  = trained["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns
                    ).sort_values(ascending=False).head(12).to_dict()

    preds_te = rf.predict(Xte)
    return dict(results=results, best_model=rf,
                features=list(X.columns), importances=imp,
                n_rows=len(df), n_features=X.shape[1],
                scatter_actual=list(yte), scatter_pred=list(preds_te))


def main():
    print("\n" + "="*58)
    print("  Student Performance Predictor — Training")
    print("  UCI Dataset · Cortez & Silva (2008)")
    print("="*58)

    A = run_experiment("Experiment A: WITH Prior Grades (clean, n=359)", clean=True,  include_prior=True)
    B = run_experiment("Experiment B: WITHOUT Prior Grades (full, n=395)", clean=False, include_prior=False)

    # Save models + feature lists
    for fname, obj in [
        ("model_with.pkl",    A["best_model"]),
        ("model_without.pkl", B["best_model"]),
        ("feats_with.pkl",    A["features"]),
        ("feats_without.pkl", B["features"]),
    ]:
        with open(os.path.join(MODELS_DIR, fname), "wb") as f:
            pickle.dump(obj, f)

    # Results JSON
    all_res = {
        "with_grades":    {k: A[k] for k in ("results","importances","features","n_rows","n_features")},
        "without_grades": {k: B[k] for k in ("results","importances","features","n_rows","n_features")},
    }
    with open(os.path.join(MODELS_DIR, "all_results.json"), "w") as f:
        json.dump(all_res, f, indent=2)

    # Scatter data for dashboard
    scatter = {
        "with_grades":    {"actual": A["scatter_actual"], "predicted": A["scatter_pred"]},
        "without_grades": {"actual": B["scatter_actual"], "predicted": B["scatter_pred"]},
    }
    with open(os.path.join(MODELS_DIR, "scatter_data.json"), "w") as f:
        json.dump(scatter, f)

    # Dataset stats
    df_full  = load_data(clean=False)
    df_clean = load_data(clean=True)
    stats = {
        "n_full":    len(df_full),
        "n_clean":   len(df_clean),
        "g3_mean":   round(float(df_clean.G3.mean()), 2),
        "g3_std":    round(float(df_clean.G3.std()),  2),
        "pass_rate": round(float((df_full.G3 >= 10).mean() * 100), 1),
        "zero_rate": round(float((df_full.G3 == 0).mean()  * 100), 1),
        "corr_g2_g3": round(float(df_clean.G2.corr(df_clean.G3)), 3),
    }
    with open(os.path.join(MODELS_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Correlation matrix for EDA
    num_cols = ["age","Medu","Fedu","traveltime","studytime","failures",
                "famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2","G3"]
    corr = df_full[num_cols].corr().round(3)
    with open(os.path.join(MODELS_DIR, "corr_matrix.json"), "w") as f:
        json.dump({"cols": num_cols, "matrix": corr.values.tolist()}, f)

    # Grade distribution
    gdist = df_full["G3"].value_counts().sort_index()
    with open(os.path.join(MODELS_DIR, "grade_dist.json"), "w") as f:
        json.dump({"grades": gdist.index.tolist(), "counts": gdist.values.tolist()}, f)

    print(f"\n✅  Saved to models/  |  Stats: {stats}\n")


if __name__ == "__main__":
    main()
