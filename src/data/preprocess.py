"""
Data loading and preprocessing pipeline for ESI triage classification.
"""
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load raw dataset from path defined in config."""
    path = Path(config["data"]["raw_path"])
    return pd.read_csv(path)


def clean_text(text: str, config: dict) -> str:
    """Apply text cleaning steps from config."""
    import re, string
    if config["preprocessing"]["lowercase"]:
        text = text.lower()
    if config["preprocessing"]["remove_punctuation"]:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def split_dataset(df: pd.DataFrame, config: dict):
    """Train / val / test stratified split."""
    from sklearn.model_selection import train_test_split
    seed = config["project"]["seed"]
    target = config["project"]["target_column"]
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]

    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target])
    adjusted_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=adjusted_val, random_state=seed, stratify=train_val[target])
    return train, val, test


if __name__ == "__main__":
    config = load_config()
    df = load_raw_data(config)
    # TODO: apply cleaning and save splits
