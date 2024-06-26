# Import necessary libraries and modules
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
from ML_pipeline.preprocessing import output_text

# Function to train a Word2Vec or FastText model and save it
def model_train(df, column_name, model, vector_size, window_size):
    # Preprocess the text data
    x = output_text(df, column_name)

    if model == 'Skipgram':
        # Train a Word2Vec Skipgram model
        skipgram = Word2Vec(x, vector_size=vector_size, window=window_size, min_count=2, sg=1)
        skipgram.save('../output/model_Skipgram.bin')  # Save the trained model to a binary file
        return skipgram  # Return the trained Word2Vec Skipgram model

    elif model == 'Fasttext':
        # Train a FastText model
        fast_text = FastText(x, vector_size=vector_size, window=window_size, min_count=2, workers=5, min_n=1, max_n=2, sg=1)
        fast_text.save('../output/model_Fasttext.bin')  # Save the trained model to a binary file
        return fast_text  # Return the trained FastText model
