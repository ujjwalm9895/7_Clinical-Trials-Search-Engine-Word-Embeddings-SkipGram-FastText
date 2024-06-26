# Import necessary libraries and modules
from numpy import dot
from numpy.linalg import norm
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
import pandas as pd
from ML_pipeline.utils import read_data
from ML_pipeline.return_embed import get_mean_vector
from ML_pipeline.preprocessing import preprocessing_input

# Define a cosine similarity function
def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Define a top_n function to return top 'n' similar results
def top_n(query, model_name, column_name):
    df = read_data("../input/Dimension-covid.csv")
    if model_name == 'Skipgram':
        word2vec_model = Word2Vec.load('../output/model_Skipgram.bin')
        K = pd.read_csv('../output/skipgram-vec-abstract.csv')
    else:
        word2vec_model = Word2Vec.load('../output/model_Fasttext.bin')
        K = pd.read_csv('../output/Fasttext-vec-abstract.csv')
    
    # Preprocess the input query
    query = preprocessing_input(query)
    query_vector = get_mean_vector(word2vec_model, query)
    
    # Load model vectors
    p = []  # Transform the dataframe into the required array-like structure
    for i in range(df.shape[0]):
        p.append(K[str(i)].values)
    
    x = []  # Convert cosine similarities of the overall dataset with input queries into a list
    for i in range(len(p)):
        x.append(cos_sim(query_vector, p[i]))
    
    # Store the list in 'tmp' to retrieve the index
    tmp = list(x)
    
    # Sort the list so that the largest elements are on the far right
    res = sorted(range(len(x)), key=lambda sub: x[sub])[-10:]
    sim = [tmp[i] for i in reversed(res)]
    
    # Get the index of the 10 or 'n' largest elements
    L = []
    for i in reversed(res):
        L.append(i)
    
    df1 = read_data("../input/Dimension-covid.csv")
    
    # Return a dataframe with only id, title, abstract, and publication date, along with the cosine similarities
    return df1.iloc[L, [1, 2, 5, 6]], sim
