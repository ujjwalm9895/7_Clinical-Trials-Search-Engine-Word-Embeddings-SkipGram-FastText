# Import necessary libraries and modules
import nltk  # The Natural Language Toolkit, used for preprocessing
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Function to calculate the mean vector for a list of words using a Word2Vec model
def get_mean_vector(word2vec_model, words):
    # Remove out-of-vocabulary words
    words = [word for word in word_tokenize(words) if word in list(word2vec_model.wv.index_to_key)]
    # If there are valid words in the list
    if len(words) >= 1:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        return np.array([0] * 100)  # Return a zero vector if no valid words are found

# Function to return word embeddings for a DataFrame column using a Word2Vec model
def return_embed(word2vec_model, df, column_name):
    K1 = []  # Initialize an empty list
    for i in df[column_name]:
        K1.append(list(get_mean_vector(word2vec_model, i)))  # Append the mean vector for each row to the list
    return K1
