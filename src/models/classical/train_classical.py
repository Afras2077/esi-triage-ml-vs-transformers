"""
Train classical ML classifiers (LR, RF, SVM, XGBoost, NB) with TF-IDF features.
"""
import joblib, yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from pathlib import Path


CLASSIFIERS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1),
    "svm": LinearSVC(class_weight="balanced", max_iter=2000),
    "xgboost": XGBClassifier(n_estimators=300, use_label_encoder=False, eval_metric="mlogloss"),
    "naive_bayes": MultinomialNB(),
}


def train_all(X_train, y_train, config: dict, out_dir: str = "models/classical/"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trained = {}
    for name, clf in CLASSIFIERS.items():
        if name in config["classical_models"]["models"]:
            clf.fit(X_train, y_train)
            path = f"{out_dir}{name}.joblib"
            joblib.dump(clf, path)
            trained[name] = clf
            print(f"[✓] {name} saved to {path}")
    return trained
