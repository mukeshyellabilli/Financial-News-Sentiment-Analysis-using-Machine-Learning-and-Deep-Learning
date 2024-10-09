from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from bs4 import BeautifulSoup
import numpy as np

app = Flask(__name__)

# Load the LSTM model and tokenizer
model = load_model('lstm_sentiment_model.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_SEQUENCE_LENGTH = 50

# Preprocessing function


def clean_text(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_sentiment():
    # Get the input data (message) from the form submission
    input_data = request.form['user_input']

    # Preprocess the input message
    cleaned_message = clean_text(input_data)
    seq = tokenizer.texts_to_sequences([cleaned_message])
    padded = pad_sequences(
        seq, maxlen=MAX_SEQUENCE_LENGTH, dtype='int32', value=0)

    # Make a prediction
    pred = model.predict(padded)

    # Map prediction to sentiment
    labels = ['Positive', 'Neutral', 'Negative']
    sentiment = labels[np.argmax(pred)]

    # Render the result in the result.html template
    return render_template('result.html', sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
