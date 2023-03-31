import warnings

import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from googletrans import Translator
from joblib import load
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from requests.exceptions import ConnectionError, Timeout, ContentDecodingError, InvalidURL, ConnectTimeout, \
    TooManyRedirects

warnings.filterwarnings("ignore")

wordnet_lemmatizer = WordNetLemmatizer()
translator = Translator()

stop = pd.read_excel(r"data\stop_words.xlsx")
stop_words = stopwords.words('english')
stop_words = set(stop_words + stop['Words'].tolist())

# ---------------------------------
# ----Load the model from disk-----
# ---------------------------------

svm = load(r"data\model.joblib")

# ---------------------------------
# ----Load TF-IDF vectorizer-------
# ---------------------------------

vectorizer = load(r"data\vectorizer.joblib")

# Create a Flask app
app = Flask(__name__)


# Define a route for the web service
@app.route('/')
def home():
    return render_template('home_page.html')


@app.route('/your_industry', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        input_url = request.form['url']

    elif request.method == 'GET':
        input_url = request.args.get('url')
    try:
        requests.get(input_url, timeout=10, headers={'Connection': 'close'})  # specify timeout
        valid = 'yes'
    except (ConnectionError, Timeout, ContentDecodingError, InvalidURL, ConnectTimeout, TooManyRedirects,
            ConnectionResetError):
        valid = 'no'

    if valid == 'yes':

        r = requests.get(input_url)
        text = r.text
        soup = BeautifulSoup(text)

        # Get the text: text

        text = soup.get_text()
        if text is None or text == 0 or len(text) <= 3:
            text = 'None'
        else:
            try:
                if detect(text) != 'en':  # check the language first
                    text = translator.translate(text, dest='en').text
                else:
                    text = text
            except LangDetectException:
                text = 'None'

        if text != 'None':
            text_new = [
                wordnet_lemmatizer.lemmatize(t.lower()) for t in word_tokenize(text)
                if t.isalpha() and t.lower() not in stop_words and len(t) >= 3
            ]

            if len(text_new) <= 5:
                result = {'result': 'Bad Quality'}
            else:
                # Transform the X_test with TF-IDF vectorizer
                ser = pd.Series(' '.join(text_new))
                x_test_tfidf = vectorizer.transform(ser)

                # Use the trained model to make predictions on the testing set
                y_pred_tfidf_svm = svm.predict(x_test_tfidf)

                result = {'result': y_pred_tfidf_svm[0]}

        else:
            result = {'result': 'Website not valid'}
    else:
        result = {'result': 'Website not valid'}
    return result


app.run()
