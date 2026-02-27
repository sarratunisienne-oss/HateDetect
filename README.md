 Narrative, Toxicity & Risk Scoring Pipeline
### Data Science Capstone Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Status](https://img.shields.io/badge/Status-In%20Progress-green)

---

## 📌 Project Overview

This project builds a reproducible data science pipeline to detect hate speech and harmful narratives in social media messages using Natural Language Processing (NLP) and Machine Learning.

The pipeline:
- Analyses text patterns and engagement behaviour
- Clusters messages into distinct narrative themes (Topic Modelling)
- Classifies content by toxicity and hate speech using transformer models
- Applies **Value Sentiment Analysis** based on Moral Foundations Theory
- Trains a predictive risk scoring model

---

## 🗂️ Project Structure

```
capstone-hate-speech/
│
├── data/
│   ├── labeled_data.csv           # Davidson Hate Speech Dataset (raw)
│   └── labeled_data_eda.csv       # Enriched dataset after EDA
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_nlp_pipeline.ipynb      # Topic Modelling + Toxicity Classification
│   ├── 03_model.ipynb             # Predictive Risk Scoring Model
│   └── 04_llm_sim.py              # LLM API Simulation Script
│
├── plots/
│   ├── plot_01_class_distribution.png
│   ├── plot_02_annotator_agreement.png
│   ├── plot_03_text_length.png
│   ├── plot_04_twitter_features.png
│   └── plot_05_correlation.png
│
├── app/
│   └── app.py                     # Streamlit Web App (hate speech detector)
│
├── reports/
│   ├── methodology_summary.md     # 400–600 word methodology & results
│   └── concept_note.md            # Policy concept note (DSA framing)
│
├── requirements.txt               # All Python dependencies
└── README.md                      # This file
```

---

## 📊 Dataset

**Source:** [Davidson Hate Speech Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)

| Property | Value |
|---|---|
| Total tweets | 24,783 |
| Language | English |
| Labels | 0 = Hate Speech, 1 = Offensive Language, 2 = Neither |
| Annotators per tweet | 3–9 human raters |

**Class Distribution:**

| Class | Count | Percentage |
|---|---|---|
| Hate Speech | 1,430 | 5.8% |
| Offensive Language | 19,190 | 77.4% |
| Neither | 4,163 | 16.8% |

> ⚠️ The dataset is **imbalanced**. Class weighting or oversampling is applied during model training.

---

## 🚀 How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com/sarratunisienne-oss/HateDetect.git
cd HateDetect
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```
Then open the notebooks in order:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_nlp_pipeline.ipynb`
3. `notebooks/03_model.ipynb`
4. `notebooks/04_llm_sim.py`

### 5. Run the web app
```bash
cd app
streamlit run app.py
```

---

## 🧪 Notebooks — What Each One Does

### 📓 01_eda.ipynb — Exploratory Data Analysis
- Loads and inspects the dataset
- Checks for missing values and duplicates
- Analyses class distribution and annotator agreement
- Engineers text features (word count, mentions, hashtags, URLs)
- Produces 5 publication-quality visualisations

**Key finding:** The dataset is heavily imbalanced (77% offensive). The hate speech annotator agreement is slightly lower than for other classes, reflecting genuine label ambiguity.

---

### 📓 02_nlp_pipeline.ipynb — NLP Pipeline
- Cleans and preprocesses tweet text
- Runs **BERTopic** to identify 15–30 narrative clusters
- Applies a **HuggingFace** pretrained toxicity classifier
- Applies **Value Sentiment Analysis** (Moral Foundations Theory)
- Scores each message on moral values: Care, Fairness, Loyalty, Authority, Purity

---

### 📓 03_model.ipynb — Predictive Risk Model
- Engineers' features from EDA + NLP outputs
- Trains baseline Logistic Regression classifier
- Trains improved Random Forest / XGBoost model
- Evaluates with cross-validation, ROC-AUC, F1, Precision-Recall
- Analyses misclassified examples

---

### 📓 04_llm_sim.py — LLM Simulation Script
- Selects the 5 most toxic or viral narrative clusters
- Designs prompt templates to extract:
  - Stance (pro/anti actor)
  - Framing (escalatory vs de-escalatory)
  - Tone (conspiratorial vs factual)
- Simulates LLM API calls (no paid calls — placeholder keys)

---

## 🌐 Web App

The Streamlit app allows anyone to paste a message and get an instant classification:

```
Input:  "your text here"
Output: ✅ Neither / ⚠️ Offensive / 🚨 Hate Speech
        + Confidence score
        + Value sentiment breakdown
```

To run:
```bash
streamlit run app/app.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
transformers
torch
bertopic
streamlit
wordcloud
langdetect
jupyter
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📅 Project Timeline

| Week | Focus | Deliverable |
|---|---|---|
| Week 1 | Data & EDA | `01_eda.ipynb` + 5 plots |
| Week 2 | NLP Pipeline | `02_nlp_pipeline.ipynb` + topic & toxicity scores |
| Week 3 | Model + LLM Sim | `03_model.ipynb` + `04_llm_sim.py` |
| Week 4 | Web App + Reports | `app.py` + concept note + README |

---

## 📋 Key Findings (Updated as project progresses)

- [ ] Temporal & engagement patterns identified
- [ ] 15–30 narrative clusters extracted
- [ ] Toxicity scores assigned to all messages
- [ ] Value sentiment scores computed
- [ ] Risk model trained and evaluated
- [ ] Web app deployed

---


## 👤 Author

**Sarra Arfaoui**  
Data Science Training — Capstone Project  
[LinkedIn](https://linkedin.com/in/sarra-arfaoui) · sarra.tunisienne@gmail.com

---

## 📄 License

This project is for educational purposes.  
Dataset credit: [Davidson et al. (2017)](https://arxiv.org/abs/1703.04009) — *Automated Hate Speech Detection and the Problem of Offensive Language*
