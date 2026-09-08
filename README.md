# 🏥 Text-Based Patient Urgency Classification

## A Comparative Study of Fine-Tuned Transformers vs. Classical Machine Learning for Emergency Department Triage

> **Research Goal:** Benchmark classical ML pipelines (TF-IDF + Logistic Regression / SVM / Random Forest / XGBoost) against fine-tuned transformer models (Bio_ClinicalBERT) on the task of predicting the **Emergency Severity Index (ESI)** level from free-text chief complaints.

---

## 📋 Table of Contents

- [Background](#-background)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Experiment Workflow](#-experiment-workflow)
- [Results](#-results)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [Citation](#-citation)

---

## 🔍 Background

Emergency Department (ED) triage is a critical bottleneck in healthcare systems worldwide. Accurate urgency classification — using the **5-level Emergency Severity Index (ESI)** — directly impacts patient outcomes. Manual triage is time-consuming and subject to inter-rater variability.

This study investigates whether **fine-tuned biomedical transformer models** (e.g., Bio_ClinicalBERT) provide a statistically significant improvement over classical bag-of-words ML pipelines for automated ESI prediction from patient chief complaint text.

**Research Questions:**
1. Can classical ML models (TF-IDF + classifier) achieve clinically acceptable ESI prediction accuracy?
2. Do transformer-based models significantly outperform classical approaches on this task?
3. What is the precision–efficiency trade-off between the two paradigms for real-world ED deployment?

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Dataset** | FEDMML ED Triage Dataset |
| **File** | `data/raw/fedmml_ed_triage_dataset.csv` |
| **Target** | ESI Level (1–5, ordinal) |
| **Primary Feature** | Free-text chief complaint |
| **Task** | Multi-class text classification |

> ⚠️ The raw CSV is excluded from version control (see `.gitignore`). Place it manually in `data/raw/` after cloning.

---

## 📁 Project Structure

```
esi-triage-ml-vs-transformers/
│
├── 📂 data/
│   ├── raw/                        # Original, immutable source data
│   │   └── fedmml_ed_triage_dataset.csv
│   ├── processed/                  # Train/val/test splits (auto-generated)
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── external/                   # Any third-party lookup tables / ICD codes
│
├── 📂 notebooks/                   # Ordered Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb   # EDA: distributions, class balance, text stats
│   ├── 02_preprocessing.ipynb      # Text cleaning, tokenisation, TF-IDF inspection
│   ├── 03_classical_ml.ipynb       # Train & evaluate classical ML baselines
│   ├── 04_transformer_finetuning.ipynb  # Fine-tune Bio_ClinicalBERT / BERT
│   └── 05_evaluation_comparison.ipynb  # Side-by-side comparison & statistical tests
│
├── 📂 src/                         # Reusable Python source modules
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py           # Data loading, cleaning, train/val/test splits
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # TF-IDF vectoriser, feature pipelines
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical/
│   │   │   ├── __init__.py
│   │   │   └── train_classical.py  # LR, RF, SVM, XGBoost, NB training
│   │   └── transformers/
│   │       ├── __init__.py
│   │       └── train_transformer.py # HuggingFace Trainer fine-tuning loop
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py             # Metrics, confusion matrix, comparison plots
│   └── utils/
│       ├── __init__.py
│       └── helpers.py              # Seed setting, config loading, logging helpers
│
├── 📂 models/                      # Serialised model artifacts (git-ignored)
│   ├── classical/                  # Saved .joblib classifiers + TF-IDF vectoriser
│   └── transformers/               # Fine-tuned checkpoint directories
│
├── 📂 results/
│   ├── figures/                    # Confusion matrices, ROC curves, bar charts
│   ├── metrics/                    # JSON / CSV metric tables per model
│   └── reports/                    # Final PDF / markdown summary reports
│
├── 📂 configs/
│   └── config.yaml                 # Central experiment configuration (paths, HPs)
│
├── 📂 tests/
│   ├── __init__.py
│   ├── test_preprocess.py          # Unit tests for data pipeline
│   ├── test_feature_engineering.py
│   └── test_evaluate.py
│
├── 📂 docs/
│   └── references/                 # Key papers (PDFs / BibTeX)
│
├── .gitignore
├── requirements.txt                # Python dependencies
└── README.md                       # You are here
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/esi-triage-ml-vs-transformers.git
cd esi-triage-ml-vs-transformers
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLP Resources

```bash
python -m nltk.downloader stopwords wordnet punkt
python -m spacy download en_core_web_sm
```

### 5. Place the Raw Dataset

Copy `fedmml_ed_triage_dataset.csv` into `data/raw/`.

---

## 🧪 Experiment Workflow

Run notebooks **in order**, or use the `src/` modules directly:

| Step | Notebook | What it does |
|------|----------|-------------|
| **1** | `01_data_exploration.ipynb` | EDA: class distribution, text length analysis, missing values |
| **2** | `02_preprocessing.ipynb` | Text cleaning, splits, TF-IDF vocabulary inspection |
| **3** | `03_classical_ml.ipynb` | Train LR / SVM / RF / XGBoost / NB; save models |
| **4** | `04_transformer_finetuning.ipynb` | Fine-tune Bio_ClinicalBERT via HuggingFace Trainer |
| **5** | `05_evaluation_comparison.ipynb` | Compare all models; statistical significance tests |

### Running via CLI (optional)

```bash
# Preprocess data
python -m src.data.preprocess

# Train classical baselines
python -m src.models.classical.train_classical

# Fine-tune transformer
python -m src.models.transformers.train_transformer
```

---

## 📈 Results

> *(To be populated after experiments are complete)*

| Model | Accuracy | Weighted F1 | Macro F1 | Cohens K |
|-------|----------|-------------|----------|-----------|
| Logistic Regression | - | - | - | - |
| SVM (LinearSVC) | - | - | - | - |
| Random Forest | - | - | - | - |
| XGBoost | - | - | - | - |
| Naive Bayes | - | - | - | - |
| Bio_ClinicalBERT | - | - | - | - |

Figures are saved to `results/figures/` and a full report to `results/reports/`.

---

## 🔧 Configuration

All experiment settings live in `configs/config.yaml`:

```yaml
transformer_models:
  base_model: "emilyalsentzer/Bio_ClinicalBERT"
  max_length: 128
  batch_size: 32
  epochs: 5
  learning_rate: 2e-5
```

Edit this file to switch models, adjust hyperparameters, or change data paths without touching source code.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-experiment`
3. Commit your changes: `git commit -m "feat: add ClinicalLongformer experiment"`
4. Push to the branch: `git push origin feature/my-experiment`
5. Open a Pull Request

---

## 📄 Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{esi-triage-ml-vs-transformers-2026,
  title   = {Text-Based Patient Urgency Classification: A Comparative Study of
             Fine-Tuned Transformer vs. Classical Machine Learning Approaches
             for Emergency Department Triage},
  author  = {<Your Name>},
  year    = {2026},
  url     = {https://github.com/<your-username>/esi-triage-ml-vs-transformers}
}
```

---

## 📚 References

- Wier, L. M., et al. (2011). Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care. AHRQ.
- Alsentzer, E., et al. (2019). Publicly Available Clinical BERT Embeddings. NAACL Clinical NLP Workshop.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL-HLT.

---

Made with love for better emergency care.
