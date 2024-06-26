#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import streamlit as st  # Importing the Streamlit library
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.decomposition import PCA
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import string
import re
import nltk
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Importing datasets
df = pd.read_csv('Dimension-covid.csv')   # Loading the dataset for preprocessing
df1 = pd.read_csv('Dimension-covid.csv')  # Loading a copy of the dataset for returning results

# Preprocessing data

# Function to remove all URLs from text
def remove_urls(text):
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return new_text

# Function to make all text lowercase
def text_lowercase(text):
    return text.lower()

# Function to remove numbers from text
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

# Function to remove punctuation from text
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Function to tokenize text
def tokenize(text):
    text = word_tokenize(text)
    return text

# Function to remove stopwords from text
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [i for i in text if not i in stop_words]
    return text

# Function to lemmatize words
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

# Combining all preprocessing steps into one function
def preprocessing(text):
    text = text_lowercase(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = ' '.join(text)
    return text

# Loading Word2Vec models
skipgram = Word2Vec.load('skipgramx11.bin')
FastText = Word2Vec.load('FastText.bin')

vector_size = 100  # Defining vector size for each word

# Function to get the mean vector of a list of words
def get_mean_vector(word2vec_model, words):
    # Remove out-of-vocabulary words
    words = [word for word in tokenize(words) if word in list(word2vec_model.wv.index_to_key)]
    if len(words) >= 1:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        return np.array([0] * 100)

# Loading precomputed vectors from CSV
K = pd.read_csv('skipgram-vec.csv')
K2 = []

for i in range(df.shape[0]):
    K2.append(K[str(i)].values)

KK = pd.read_csv('FastText-vec.csv')
K1 = []

for i in range(df.shape[0]):
    K1.append(KK[str(i)].values)

# Function to calculate cosine similarity between two vectors
def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Setting options for displaying full text in the dataframe
pd.set_option("display.max_colwidth", -1)

# Streamlit function
def main():
    # Load data and models
    data = df1  # The dataset we want to display

    st.title("Clinical Trial Search Engine")  # Title of our app
    st.write('Select Model')  # Text below the title

    Vectors = st.selectbox("Model", options=['Skipgram', 'Fasttext'])
    if Vectors == 'Skipgram':
        K = K2
        word2vec_model = skipgram
    elif Vectors == 'Fasttext':
        K = K1
        word2vec_model = FastText

    st.write('Type your query here')

    query = st.text_input("Search box")  # Getting input from the user

    def preprocessing_input(query):
        query = preprocessing(query)
        query = query.replace('\n', ' ')
        K = get_mean_vector(word2vec_model, query)
        return K

    def top_n(query, p, df1):
        query = preprocessing_input(query)
        x = []

        for i in range(len(p)):
            x.append(cos_sim(query, p[i]))
        tmp = list(x)
        res = sorted(range(len(x), key=lambda sub: x[sub])[-10:])
        sim = [tmp[i] for i in reversed(res)]

        L = []
        for i in reversed(res):
            L.append(i)
        return df1.iloc[L, [1, 2, 5, 6]], sim

    model = top_n
    if query:
        P, sim = model(str(query), K, data)
        # Plotly function to display our dataframe as a plotly table
        fig = go.Figure(data=[go.Table(header=dict(values=['ID', 'Title', 'Abstract', 'Publication Date', 'Score']),
                                      cells=dict(values=[list(P['Trial ID'].values), list(P['Title'].values),
                                                         list(P['Abstract'].values), list(P['Publication date'].values),
                                                         list(np.around(sim, 4))], align=['center', 'right']))])
        # Displaying the plotly table
        fig.update_layout(height=1700, width=700, margin=dict(l=0, r=10, t=20, b=20))

        st.plotly_chart(fig)

# Entry point for the application
if __name__ == "__main__":
    main()
