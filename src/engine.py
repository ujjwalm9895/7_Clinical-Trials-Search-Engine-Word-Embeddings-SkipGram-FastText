# Import necessary libraries and modules
from ML_pipeline.utils import read_data
from ML_pipeline.preprocessing import output_text
from ML_pipeline.train_model import model_train
from ML_pipeline.return_embed import return_embed
from ML_pipeline.top_n import top_n
import pandas as pd
from gensim.models import Word2Vec

# Read the initial dataset from two separate paths
df = read_data("../input/Dimension-covid.csv")
df1 = read_data("../input/Dimension-covid.csv")  

# Pre-process the data and extract text from the "Abstract" column
x = output_text(df, "Abstract")

# Train the Skipgram and FastText models on the "Abstract" text
skipgram = model_train(df, "Abstract", "Skipgram", vector_size=100, window_size=1)
fasttext = model_train(df, "Abstract", "Fasttext", vector_size=100, window_size=2)

# Load the pretrained Word2Vec models
skipgram = Word2Vec.load('../output/model_Skipgram.bin')
FastText = Word2Vec.load('../output/model_Fasttext.bin')

# Convert the text data to column vectors using Skipgram
K1_abstract = return_embed(skipgram, df, "Abstract")
K1_abstract = pd.DataFrame(K1_abstract).transpose()
K1_abstract.to_csv('../output/skipgram-vec-abstract.csv')

K1_title = return_embed(skipgram, df, "Title")
K1_title = pd.DataFrame(K1_title).transpose()
K1_title.to_csv('../output/skipgram-vec-title.csv')

# Convert the text data to column vectors using FastText
K2_abstract = return_embed(FastText, df, "Abstract")
K2_abstract = pd.DataFrame(K2_abstract).transpose()
K2_abstract.to_csv('../output/FastText-vec-abstract.csv')

K2_title = return_embed(FastText, df, "Title")
K2_title = pd.DataFrame(K2_title).transpose()
K2_title.to_csv('../output/FastText-vec-title.csv')

# Load the pretrained vectors of each abstract from the saved CSV
K = read_data('../output/skipgram-vec-abstract.csv')
skipgram_vectors = []  # Initialize an empty list to store vectors

# Transform the dataframe into an array-like structure
for i in range(df.shape[0]):
    skipgram_vectors.append(K[str(i)].values)

# Use the 'top_n' function to return 'n' similar results for a given query
Results = top_n('Coronavirus', 'Skipgram', 'Abstract')
