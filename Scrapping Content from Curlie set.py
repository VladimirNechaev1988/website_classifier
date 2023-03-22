import pandas as pd
import requests
from requests.exceptions import ConnectionError, Timeout, ContentDecodingError, InvalidURL, ConnectTimeout, \
    TooManyRedirects, SSLError

from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from csv import reader
import re
import warnings
from googletrans import Translator
from nltk.corpus import stopwords
from tldextract import extract

warnings.filterwarnings("ignore")
translator = Translator()

df = pd.DataFrame(columns=['Full_URL', 'Domain', 'Content', 'Language'])


def scrapper(url):
    try:
        requests.get(url, timeout=10.0,
                     headers={'Connection': 'close'})  # specify timeout
        global valid
        valid = 'yes'  # assign the value for the global variable

    except (ConnectionError, Timeout, ContentDecodingError, InvalidURL,
            ConnectTimeout, TooManyRedirects,
            ConnectionResetError, SSLError):  # no connection error (includes SSL error)
        valid = 'no'

    if valid == 'no':
        print('Website', url, 'does not exist')
        df.loc[len(df)] = [
            url, 'Website does not exist', 'Website does not exist', 'Website does not exist'
        ]
    else:
        print('Website', url, 'is valid')

        r = requests.get(url)
        text = r.text
        soup = BeautifulSoup(text)

        # Get the text: text

        text = soup.get_text()
        if text is None or text == 0 or len(text) <= 3:
            text = 'None'
            lang = 'None'
        else:
            try:
                lang = detect(text)
                if detect(text) != 'en':  # check the language first
                    text = translator.translate(text, dest='en').text
                else:
                    text = text
            except LangDetectException:
                text = 'None'
                lang = 'None'

        # Get the domain

        tsd, td, tsu = extract(url)
        domain = re.sub('[^a-zA-Z0-9]', '', td)

        df.loc[len(df)] = [
            url, domain, text, lang
        ]
        df.to_excel(r"C:\Users\Lumitos\Desktop\Table_curlie_test.xlsx")


data = pd.read_excel(
    r"C:\Users\Lumitos\OneDrive - IU International University of Applied Sciences\IUBH учеба\Thesis\Data "
    r"Sets\KNN\Test_set.xlsx",
)

for index, row in data[['URL']].iterrows():
    scrapper(row.squeeze())
