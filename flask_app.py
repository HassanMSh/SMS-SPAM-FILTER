# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, jsonify
import pickle

# Import necessary libraries

from operator import index
import pandas as pd
import numpy as np
import os
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download necessary data for natural language processing tasks

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

app = Flask(__name__)

API_KEYS = {
    'developer1': 'whatever',
    'developer2': 'whatever2'
}

# Load the pickle file
with open('/home/hms/mysite/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the pickle file
with open('/home/hms/mysite/cv.pkl', 'rb') as f:
    cv = pickle.load(f)

def preprocess_data(df):
    wordnet_lem = WordNetLemmatizer()

    df = pd.DataFrame([{'text': df[0]['text']}])
    reg_vars = ['http\S+', 'www\S+', 'https\S+', '\W\s+', '\d+', '\t+', '\d+', '\-+', '\\+', '\/+', '\"+', '\#+', '\++', '\@+', '\$+',  '\%+', '\^+', '\&+', '\*+', '\(+', '\)+', '\[+', '\]+', '\{+', '\}+', '\|+', '\;+', '\:+', '\<+', '\>+', '\?+', '\,+', '\.+', '\=+',     '\_+', '\~+', '\`+', '\s+']
    df['text'].replace(reg_vars, ' ', regex=True, inplace=True)
    df['text'] = df['text'].astype(str).str.lower()
    df = df[df['text'].map(lambda x: x.isascii())]
    df['text'] = df.apply(lambda column: nltk.word_tokenize(column['text']), axis=1)
    stopwords = nltk.corpus.stopwords.words('english')
    df['text'] = df['text'].apply(lambda x: [item for item in x if item not in stopwords])
    df['text'] = df['text'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
    df['text'] = df['text'].apply(wordnet_lem.lemmatize)

    processed_data = cv.transform(df['text']).toarray()

    return processed_data

@app.route('/predict', methods=['POST'])
def predict():

    api_key = request.headers.get('API-Key')

    if api_key is not None and api_key in API_KEYS.values():

        # Get the input data from the request
        data = request.get_json()
        data = [data]

        # Preprocess the data
        processed_data = preprocess_data(data)

        # Use the model to make a prediction
        prediction = model.predict(processed_data)

        # Convert the NumPy array to a Python list
        prediction_list = prediction.tolist()

        # Return "ham" or "spam" depending on the prediction
        if prediction_list[0] == 0:
            return jsonify({'prediction': 'Not Spam'})
        else:
            return jsonify({'prediction': 'Spam'})
    else:
        # API key is not valid, return an error
        return jsonify({'error': 'Invalid API key'}), 401

    # Return the prediction in the response
    return jsonify({'prediction': prediction})