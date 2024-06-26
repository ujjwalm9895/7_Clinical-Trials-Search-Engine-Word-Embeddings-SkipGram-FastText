# Import necessary libraries and modules
import string  # Used for preprocessing
import re  # Used for preprocessing
import nltk  # The Natural Language Toolkit, used for preprocessing
import numpy as np  # Used for managing NaNs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  # Used for preprocessing
from nltk.stem import WordNetLemmatizer  # Used for preprocessing

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Function to remove all URLs from text
def remove_urls(text):
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return new_text

# Make all text lowercase
def text_lowercase(text):
    return text.lower()

# Remove numbers from text
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

# Remove punctuation from text
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Tokenize text
def tokenize(text):
    text = word_tokenize(text)
    return text

# Remove stopwords from text
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [i for i in text if not i in stop_words]
    return text

# Lemmatize words
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

# Create one function to apply all preprocessing steps at once
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

# Apply preprocessing and remove '\n' character
def output_text(df, column_name):
    for i in range(df.shape[0]):
        df[column_name][i] = preprocessing(str(df[column_name][i]))
    for text in df[column_name]:
        text = text.replace('\n', ' ')
    x = [word_tokenize(word) for word in df[column_name]]  # Tokenize the data for training purposes
    return x

# Preprocess input text to match the training data format
def preprocessing_input(query):
    query = preprocessing(query)
    query = query.replace('\n', ' ')
    return query
