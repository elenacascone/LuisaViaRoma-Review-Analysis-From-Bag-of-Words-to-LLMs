# LuisaViaRoma Review Analysis: From Bag of Words to LLMs

🇮🇹 *[Leggi la versione Italiana](README.md)*

**Sentiment Analysis & Topic Modeling on Luxury E-Commerce Reviews — from VADER to Fine-Tuned BERT**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-informational)
![BERT](https://img.shields.io/badge/Deep%20Learning-BERT%20Fine--Tuning-orange)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

---

## Executive Summary

A luxury e-commerce platform (**LuisaViaRoma**) receives thousands of textual reviews on Trustpilot; manually reviewing them to comprehend shortcomings in the customer experience is a highly unscalable approach.

This project collects **approximately 10,000 authentic reviews** via web scraping (Selenium), translates and thoroughly processes them, and subsequently compares **6 sentiment analysis approaches**. These range from a rule-based baseline (VADER), to classical Machine Learning models leveraging TF-IDF (Naive Bayes, Decision Tree, Random Forest, SVM), up to the **fine-tuning of BERT**. Furthermore, a **Topic Modeling (LDA)** module is utilized to isolate the operational "pain points" concealed within the negative reviews.

**Principal Outcome:** The fine-tuning of BERT elevates the Macro F1-Score from 0.62 (achieved by the best classical ML model, SVM) to **0.73**. This quantitatively demonstrates the superiority of contextual Deep Learning over Bag-of-Words representations in accurately capturing sarcasm, negations, and minority classes (such as "neutral" reviews).

---

## Project Objectives

To develop a comprehensive NLP pipeline capable of:
1. Acquiring real-world data "in the wild" via web scraping.
2. Automatically classifying review sentiment (Positive / Neutral / Negative).
3. Extracting latent themes (topics) underlying negative feedback.
4. Rigorously comparing lexicon-based, classical ML, and Deep Learning methodologies, quantifying the authentic "added value" provided by each.

## Project Pipeline

| Phase | Description |
|------|-------------|
| **1. Data Acquisition** | Web scraping of Trustpilot reviews for LuisaViaRoma using Selenium, followed by translation into English. |
| **2. Data Cleaning** | Removal of personal data (author names), management of missing values, and temporal filtering (post-2019 to mitigate statistical noise). |
| **3. EDA** | Analysis of temporal distribution, star rating distribution, review length per rating, geographical patterns, and quality trends over time. |
| **4. NLP Pre-processing** | Lowercasing, regex cleaning, custom stop-words implementation (preserving critical negations), lemmatization (WordNet), application of a domain-specific synonym dictionary, and brand stop-word removal. |
| **5. Text Classification (ML)** | Application of TF-IDF (uni+bigrams) alongside Naive Bayes, Decision Tree, Random Forest, and Linear SVM; management of class imbalance utilizing **SMOTE**. |
| **6. Topic Modeling** | Implementation of LDA on negative reviews, selecting the optimal number of topics (K=5) via **UMass Coherence** evaluation. |
| **7. Deep Learning** | Fine-tuning of **BERT** (`bert-base-uncased`) for robust 3-class classification. |
| **8. Lexicon-Based Baseline** | Counterfactual comparison against **VADER** to ascertain the analytical ROI of Deep Learning. |
| **9. Comparative Synthesis** | Final benchmarking of all evaluated models based on their **Macro F1-Score**. |

## Key Insights from EDA

- The rating distribution is heavily skewed toward 5 stars (a J-curve phenomenon typical of online reviews).
- Negative reviews are significantly lengthier than positive ones (concise praise versus detailed, specific complaints).
- No geographical market records an average below 3.5 stars, although distinct regional patterns emerge within lower-scoring territories.
- Feature importance analysis (TF-IDF + Random Forest) indicates that negations ("not", "no") function as the most potent predictors of sentiment.

## Models and Results

### Macro F1-Score Comparison (Final Metric)

| Model | Type | Macro F1 |
|---|---|---|
| VADER (Lexicon-based) | Rule-based baseline | 0.56 |
| Decision Tree | Classical ML | 0.56 |
| Naive Bayes | Classical ML | 0.60 |
| Random Forest | Classical ML | 0.58 |
| Linear SVM | Classical ML | 0.62 |
| **BERT (fine-tuned)** | **Deep Learning** | **0.73** 🏆 |

> The **Macro F1-Score** was selected as the comparative metric (rather than global Accuracy) because, in the presence of severe class imbalance, it assigns equal weight to each category. This prevents models from being erroneously rewarded for merely predicting the majority class (5 stars).

**Why BERT Prevails:** Unlike Bag-of-Words methodologies, BERT captures the bidirectional context of text. It successfully navigates sarcasm, double negations, and highly nuanced reviews (e.g., "Great job losing my package") that typically confound both VADER and linear classifiers relying upon TF-IDF.

### Topic Modeling (LDA, K=5, negative reviews)

Five latent topics were successfully identified within the negative reviews (selected via UMass Coherence optimization). These include a dominant cluster associated with procedural friction regarding returns (keywords: *parcel, return, tag, company*) and another directly pertaining to customer service issues (*customer, service, refund*).

## 🛠️ Tech Stack

**Language:** Python 3.10+

**Primary Libraries:**
- **Data manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **Classical NLP:** `nltk`, `langdetect`, `scikit-learn` (`TfidfVectorizer`)
- **Machine Learning:** `scikit-learn` (Naive Bayes, Decision Tree, Random Forest, SVM/LinearSVC), `imbalanced-learn` (SMOTE)
- **Topic Modeling:** `gensim` (LDA, Coherence Model)
- **Deep Learning:** `torch`, `transformers`, `datasets` (fine-tuning BERT)
- **Sentiment lexicon-based:** `vaderSentiment`
- **Web scraping (data collection):** `selenium`

## 📁 Repository Structure

```
.
├── ENGLISH VERSION
|        └── Project LVM.ipynb     # Comprehensive notebook containing the entire pipeline ENGLISH VERSION
|        └── Project LVM.pdf.      # PDF of the executed notebook ENGLISH VERSION
├── Progetto_completo.ipynb        # Comprehensive notebook containing the entire pipeline ITALIAN VERSION
├── Progetto_completo.pdf          # PDF of the executed notebook ITALIAN VERSION
└── README.md
```

> ⚠️ Privacy Note: The column containing the review authors' names was removed from the dataset prior to publication, as it constitutes authentic personal data that is strictly unnecessary for the analysis.

**THE NOTEBOOK IS NOT EXECUTABLE BECAUSE, DUE TO STRICT PRIVACY AND LEGAL CONSTRAINTS, IT IS NOT PERMISSIBLE TO UPLOAD THE DATASET TO A PUBLIC PLATFORM. WE, THE AUTHORS, OPT TO RETAIN THE DATASET LOCALLY.**

## Future Developments

- Deployment of the BERT model as a dedicated API/service for the real-time classification of incoming reviews.
- Expansion of the Topic Modeling phase utilizing more contemporary techniques (e.g., BERTopic) to facilitate a direct comparison with LDA.
- Native multilingual analysis, thereby bypassing the intermediate translation phase entirely.

## 👩‍💻 Authors

**Elena Cascone**
Junior Data Scientist

- 🔗 GitHub: [github.com/elenacascone](https://github.com/elenacascone)
- 💼 LinkedIn: [linkedin.com/in/elena-cascone-18ec](https://www.linkedin.com/in/elena-cascone-18ec/)

**Alessandra Cosentino**
Junior Data Scientist

- 🔗 GitHub: [github.com/alessandracosent7-hash](https://github.com/alessandracosent7-hash)
- 💼 LinkedIn: [linkedin.com/in/alessandra-cosentino](https://www.linkedin.com/in/alessandra-cosentino/)
