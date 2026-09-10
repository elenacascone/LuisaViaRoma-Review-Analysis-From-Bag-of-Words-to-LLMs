# Analisi delle recensioni di LuisaViaRoma: dal Bag of Words agli LLM

**Sentiment Analysis & Topic Modeling su recensioni e-commerce di lusso — da VADER a BERT fine-tuned**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-informational)
![BERT](https://img.shields.io/badge/Deep%20Learning-BERT%20Fine--Tuning-orange)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

---

## Executive Summary

Un e-commerce di lusso (**LuisaViaRoma**) riceve migliaia di recensioni testuali su Trustpilot: leggerle manualmente per capire cosa non funziona nella customer experience non è scalabile.

Questo progetto raccoglie **~10.000 recensioni reali** tramite web scraping (Selenium), le traduce e pulisce, e confronta **6 approcci di sentiment analysis** — da una baseline a regole (VADER), a modelli ML classici su TF-IDF (Naive Bayes, Decision Tree, Random Forest, SVM), fino al **fine-tuning di BERT**. Un modulo di **Topic Modeling (LDA)** viene poi usato per isolare i "pain point" operativi nascosti nelle recensioni negative.

**Risultato principale:** il fine-tuning di BERT porta il Macro F1-Score da 0.62 (miglior modello ML classico, SVM) a **0.73**, dimostrando in modo quantitativo il valore del Deep Learning contestuale rispetto alle rappresentazioni Bag-of-Words nel cogliere sarcasmo, negazioni e classi minoritarie (recensioni "neutre").

---

## Obiettivo del progetto

Sviluppare una pipeline completa di NLP in grado di:
1. Raccogliere dati reali "in the wild" tramite web scraping
2. Classificare automaticamente il sentiment delle recensioni (Positivo / Neutro / Negativo)
3. Estrarre i temi latenti (topic) alla base delle recensioni negative
4. Confrontare rigorosamente approcci lexicon-based, ML classici e Deep Learning, quantificando il reale "valore aggiunto" di ciascuno

## Pipeline del progetto

| Fase | Descrizione |
|------|-------------|
| **1. Data Acquisition** | Web scraping con Selenium delle recensioni Trustpilot di LuisaViaRoma; traduzione in inglese |
| **2. Data Cleaning** | Rimozione dati personali (nomi autori), gestione valori mancanti, filtraggio temporale (post-2019 per ridurre rumore statistico) |
| **3. EDA** | Distribuzione temporale, distribuzione delle stelle, lunghezza recensioni per rating, analisi geografica, trend qualità nel tempo |
| **4. Pre-processing NLP** | Lowercasing, pulizia regex, stop-words custom (con preservazione delle negazioni), lemmatizzazione (WordNet), dizionario di sinonimi di dominio, rimozione stop-words del brand |
| **5. Text Classification (ML)** | TF-IDF (uni+bigrammi) + Naive Bayes, Decision Tree, Random Forest, SVM lineare; gestione dello sbilanciamento delle classi con **SMOTE** |
| **6. Topic Modeling** | LDA sulle recensioni negative, selezione del numero ottimale di topic (K=5) via **UMass Coherence** |
| **7. Deep Learning** | Fine-tuning di **BERT** (`bert-base-uncased`) per classificazione a 3 classi |
| **8. Baseline Lexicon-Based** | Confronto controfattuale con **VADER** per quantificare il ROI analitico del Deep Learning |
| **9. Sintesi comparativa** | Confronto finale di tutti i modelli su **Macro F1-Score** |

## Principali insight dall'EDA

- Distribuzione fortemente sbilanciata verso le 5 stelle (curva a J tipica delle recensioni online)
- Le recensioni negative sono significativamente più lunghe di quelle positive (elogi concisi vs. lamentele dettagliate)
- Nessun mercato geografico scende sotto le 3.5 stelle medie, ma emergono pattern regionali nei mercati con punteggi più bassi
- Analisi della feature importance (TF-IDF + Random Forest) rivela che le negazioni ("not", "no") sono i predittori più potenti del sentiment

## Modelli e risultati

### Confronto Macro F1-Score (metrica finale)

| Modello | Tipo | Macro F1 |
|---|---|---|
| VADER (Lexicon-based) | Baseline a regole | 0.56 |
| Decision Tree | ML classico | 0.56 |
| Naive Bayes | ML classico | 0.60 |
| Random Forest | ML classico | 0.58 |
| SVM Lineare | ML classico | 0.62 |
| **BERT (fine-tuned)** | **Deep Learning** | **0.73** 🏆 |

> Si è scelto il **Macro F1-Score** come metrica di confronto (anziché l'Accuracy globale) perché, in presenza di forte sbilanciamento delle classi, assegna pari peso a ogni categoria — evitando che i modelli vengano premiati semplicemente per aver "indovinato" la classe maggioritaria (5 stelle).

**Perché BERT vince:** a differenza degli approcci Bag-of-Words, BERT coglie il contesto bidirezionale del testo, riuscendo a gestire correttamente sarcasmo, doppie negazioni e recensioni sfumate (es. "Great job losing my package") che ingannano sia VADER sia i classificatori lineari su TF-IDF.

### Topic Modeling (LDA, K=5, recensioni negative)

Individuati 5 topic latenti nelle recensioni negative (selezionati tramite ottimizzazione della UMass Coherence), tra cui un cluster dominante legato all'attrito procedurale sui resi (parole chiave: *parcel, return, tag, company*) e uno legato al servizio clienti (*customer, service, refund*).

## 🛠️ Tech stack

**Linguaggio:** Python 3.10+

**Librerie principali:**
- **Data manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **NLP classico:** `nltk`, `langdetect`, `scikit-learn` (`TfidfVectorizer`)
- **Machine Learning:** `scikit-learn` (Naive Bayes, Decision Tree, Random Forest, SVM/LinearSVC), `imbalanced-learn` (SMOTE)
- **Topic Modeling:** `gensim` (LDA, Coherence Model)
- **Deep Learning:** `torch`, `transformers`, `datasets` (fine-tuning BERT)
- **Sentiment lexicon-based:** `vaderSentiment`
- **Web scraping (raccolta dati):** `selenium`

## 📁 Struttura del repository

```
.
├── Progetto_completo.ipynb        # notebook completo con l'intera pipeline
├── Progetto_completo.pdf          #pdf del notebook con il codice già runnato
└── README.md
```

> ⚠️ Nota privacy: la colonna con i nomi degli autori delle recensioni è stata rimossa dal dataset prima della pubblicazione, in quanto dato personale reale non necessario all'analisi.

**IL NOTEBOOK NON È ESEGUIBILE PERCHÈ PER PROBLEMI DI PRIVACY E LEGALI NON è POSSIBILE CARICARE IL DATASET SU UNA PIATTAFORMA PUBBLICA, NOI AUTRICI PREFERIAMO MANTENERE IL DATASET IN LOCALE**

## Possibili sviluppi futuri

- Deploy del modello BERT come API/servizio per la classificazione in tempo reale di nuove recensioni
- Estensione del Topic Modeling con tecniche più recenti (es. BERTopic) per un confronto diretto con LDA
- Analisi multilingue nativa, senza passaggio intermedio di traduzione

## 👩‍💻 Autrici

**Elena Cascone**
Laureanda in Data Science (Università degli Studi di Napoli Federico II) — Junior Data Scientist

- 🔗 GitHub: [github.com/elenacascone](https://github.com/elenacascone)
- 💼 LinkedIn: [linkedin.com/in/elena-cascone-18ec](https://www.linkedin.com/in/elena-cascone-18ec/)

**Alessandra Cosentino**
Laureanda in Data Science (Università degli Studi di Napoli Federico II) — Junior Data Scientist

- 🔗 GitHub: [github.com/alessandracosent7-hash](https://github.com/alessandracosent7-hash)
- 💼 LinkedIn: [linkedin.com/in/alessandra-cosentino](https://www.linkedin.com/in/alessandra-cosentino/)

