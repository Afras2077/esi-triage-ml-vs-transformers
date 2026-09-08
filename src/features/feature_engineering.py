"""
Feature engineering: TF-IDF vectorisation and classical ML feature extraction.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib, yaml
from pathlib import Path


def build_tfidf(config: dict):
    cfg = config["classical_models"]["tfidf"]
    return TfidfVectorizer(
        max_features=cfg["max_features"],
        ngram_range=tuple(cfg["ngram_range"]),
        sublinear_tf=cfg["sublinear_tf"],
    )


def fit_and_save_tfidf(X_train, config: dict, out_path: str = "models/classical/tfidf.joblib"):
    vectorizer = build_tfidf(config)
    vectorizer.fit(X_train)
    joblib.dump(vectorizer, out_path)
    return vectorizer
