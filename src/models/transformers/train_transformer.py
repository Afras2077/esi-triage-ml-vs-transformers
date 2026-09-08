"""
Fine-tune a pre-trained transformer (Bio_ClinicalBERT) for ESI classification.
"""
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
import torch, yaml
from pathlib import Path


def get_tokenizer_and_model(config: dict, num_labels: int):
    model_name = config["transformer_models"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


def tokenize_dataset(df, tokenizer, config: dict, text_col: str):
    max_len = config["transformer_models"]["max_length"]
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset.map(
        lambda x: tokenizer(x[text_col], truncation=True, padding="max_length", max_length=max_len),
        batched=True,
    )


def get_training_args(config: dict) -> TrainingArguments:
    cfg = config["transformer_models"]
    return TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="mlflow",
    )
