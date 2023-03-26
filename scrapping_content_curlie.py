import pandas as pd
import requests
from requests.exceptions import ConnectionError, Timeout, ContentDecodingError, InvalidURL, ConnectTimeout, \
    TooManyRedirects, SSLError
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
import re
import warnings
from googletrans import Translator
from tldextract import extract

warnings.filterwarnings("ignore")
translator = Translator()
valid=''

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
        # Save the results of the test set (alternate with train)
        # df.to_excel(r"data\test_scrapped_init.xlsx")
        # df.to_excel(r"data\training_scrapped_init.xlsx")


# Open the test URLs (alternate with train)
data = pd.read_excel(r"data\URL_test.xlsx")
# data = pd.read_excel(r"data\URL_training.xlsx")


for index, row in data[['URL']].iterrows():
    scrapper(row.squeeze())
