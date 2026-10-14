"""Train, compare and save churn models.  Run:  python src/train.py"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import TARGET, build_preprocessor, load_and_clean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telco_churn.csv"
SEED = 42


def main():
    df = load_and_clean(DATA)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Rows: {len(df)} | churn rate: {y.mean():.1%}")

    models = {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="logloss", random_state=SEED,
        ),
    }

    results, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        results.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, proba),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        })
        fitted[name] = pipe

    res = pd.DataFrame(results).sort_values("roc_auc", ascending=False).round(3)
    print("\n", res.to_string(index=False))
    (ROOT / "reports").mkdir(exist_ok=True)
    res.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

    # Pick the best real model (exclude baseline) by ROC-AUC
    best_name = res[res.model != "Baseline (majority class)"].iloc[0]["model"]
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": best, "columns": X.columns.tolist(), "name": best_name},
                ROOT / "models" / "churn_model.joblib")

    # ---- SHAP global explanation ----
    prep, clf = best.named_steps["prep"], best.named_steps["clf"]
    Xt = pd.DataFrame(prep.transform(X_test), columns=prep.get_feature_names_out())
    background = pd.DataFrame(prep.transform(X_train), columns=Xt.columns).sample(200, random_state=SEED)
    explainer = shap.Explainer(clf, background)
    shap_values = explainer(Xt.iloc[:500])
    if shap_values.values.ndim == 3:  # some models return one slice per class
        shap_values = shap_values[:, :, 1]
    shap.summary_plot(shap_values, Xt.iloc[:500], show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "shap_summary.png", dpi=150, bbox_inches="tight")
    print("Saved reports/shap_summary.png")


if __name__ == "__main__":
    main()
