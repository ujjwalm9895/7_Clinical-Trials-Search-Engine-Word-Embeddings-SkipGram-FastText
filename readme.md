# Medical Search Engine using Word2Vec and FastText with Gensim

## Business Objective
In the domain of Natural Language Processing (NLP), extracting context from text data is a significant challenge. Word embeddings, which represent words as semantically meaningful dense vectors, address many limitations of other techniques such as one-hot encodings and TFIDF. They enhance generalization and performance in downstream NLP applications, even with limited data. Word embedding is a feature learning technique that maps words or phrases from the vocabulary to real-number vectors, capturing contextual relationships.

General word embeddings may not perform optimally across all domains. Therefore, this project focuses on creating domain-specific medical word embeddings using Word2Vec and FastText in Python. Word2Vec is a combination of models for distributed word representations, while FastText is an efficient library for learning word representations and sentence classification developed by the Facebook Research Team.

The project's ultimate goal is to use the trained models (Word2Vec and FastText) to build a search engine and a Streamlit user interface.

---

## Data Description
For this project, we are using a clinical trials dataset related to Covid-19. You can access the dataset [here](https://dimensions.figshare.com/articles/dataset/Dimensions_COVID-19_publications_datasets_and_clinical_trials/11961063). The dataset comprises 10,666 rows and 21 columns, with the following two essential columns:
- Title
- Abstract

---

## Aim
The project's objective is to train Skip-gram and FastText models to perform word embeddings and then build a search engine for clinical trials dataset with a Streamlit user interface.

---

## Tech Stack
- **Language**: `Python`
- **Libraries and Packages**: `pandas`, `numpy`, `matplotlib`, `plotly`, `gensim`, `streamlit`, `nltk`

---

## Approach
1. Import the required libraries.
2. Read the dataset.
3. Data preprocessing:
   - Remove URLs
   - Convert text to lowercase
   - Remove numerical values
   - Remove punctuation
   - Tokenization
   - Remove stop words
   - Lemmatization
   - Remove '\n' character from the columns.
4. Exploratory Data Analysis (EDA):
   - Data Visualization using word cloud.
1. Train the 'Skip-gram' model.
2. Train the 'FastText' model.
3. Model embeddings - Similarity.
4. PCA plots for Skip-gram and FastText models.
5. Convert abstract and title to vectors using the Skip-gram and FastText model.
6. Use the Cosine similarity function.
7. Perform input query pre-processing.
8. Define a function to return the top 'n' similar results.
9. Result evaluation.
10. Run the Streamlit Application.

---

## Modular Code Overview

1. **input**: Contains the data used for analysis, a clinical trials dataset based on Covid-19 (`Dimension-covid.csv`).

2. **src**: This is the most important folder and contains modularized code for all the steps in a modularized manner. It includes:
   - `engine.py`
   - `ML_pipeline`: A folder with functions split into different Python files, which are appropriately named. These functions are called within `engine.py`.

3. **output**: Contains the best-fitted model trained on this data, which can be easily loaded and used for future applications without the need to retrain the models from scratch.

4. **lib**: A reference folder with:
   - The original iPython notebook.
   - The `Medical.py` notebook for running the Streamlit UI.

---
