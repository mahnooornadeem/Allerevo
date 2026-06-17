import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
import joblib

CSV = "allergen_clean.csv"   # same folder as this script

def load():
    df = pd.read_csv(CSV)
    df["identity"] = pd.to_numeric(df["identity"], errors="coerce")
    df["length"]   = pd.to_numeric(df["length"], errors="coerce")
    df["has_homolog"] = df["identity"].notna().astype(int)
    df["is_plant"]    = df["kingdom"].str.lower().str.startswith("plant").astype(int)
    df["label"] = (df["status"] == "Yes").astype(int)   # 1 = allergen
    return df

def evaluate(name, X, y):
    """Run 5-fold CV for RF and SVM and print honest metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1"]
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVM (RBF)": Pipeline([("scale", StandardScaler()),
                               ("svc", SVC(kernel="rbf", random_state=42))]),
    }
    print(f"\n{'='*64}\n{name}  (n={len(y)}, allergens={int(y.sum())}, non={int((1-y).sum())})\n{'='*64}")
    for mname, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring)
        print(f"\n  {mname}")
        for s in scoring:
            vals = res["test_" + s]
            print(f"    {s:10s}: {vals.mean():.3f}  (+/- {vals.std():.3f})")
    # baseline: always guess the majority class
    base = max(y.mean(), 1 - y.mean())
    print(f"\n  Majority-class baseline accuracy: {base:.3f}  (beat this to be meaningful)")

def feature_importance(X, y, cols):
    rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, y)
    order = np.argsort(rf.feature_importances_)[::-1]
    print("\n  Feature importance (Random Forest):")
    for i in order:
        print(f"    {cols[i]:14s}: {rf.feature_importances_[i]:.3f}")
    return rf

def main():
    df = load()

    # ---- Experiment A: proteins WITH a human homolog ----
    a = df[df["has_homolog"] == 1].copy()
    colsA = ["identity", "length", "is_plant"]
    Xa = a[colsA].fillna(a[colsA].median()).values
    ya = a["label"].values
    evaluate("EXPERIMENT A - proteins with a human homolog", Xa, ya)
    feature_importance(Xa, ya, colsA)

    # ---- Experiment B: full dataset ----
    b = df.copy()
    # impute missing identity with the median of those that have it,
    # and let has_homolog carry the 'no human counterpart' signal
    med = b.loc[b.has_homolog == 1, "identity"].median()
    b["identity_f"] = b["identity"].fillna(med)
    colsB = ["identity_f", "has_homolog", "length", "is_plant"]
    Xb = b[colsB].fillna(b[colsB].median()).values
    yb = b["label"].values
    evaluate("EXPERIMENT B - full dataset (no-homolog = most diverged)", Xb, yb)
    rf = feature_importance(Xb, yb, colsB)

    # save the full-data Random Forest
    joblib.dump({"model": rf, "features": colsB}, "allergen_model.pkl")
    print("\nSaved trained model -> allergen_model.pkl")

if __name__ == "__main__":
    main()
