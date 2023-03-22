import pandas as pd
import requests
import treetaggerwrapper as treetaggerwrapper
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

stop_words_en = stopwords.words('english')
# stop_words_de = stopwords.words('german')
translator = Translator()
# tagger_de = treetaggerwrapper.TreeTagger(TAGLANG='de')
tagger_en = treetaggerwrapper.TreeTagger(TAGLANG='en')
df = pd.DataFrame(columns=['Full_URL', 'Content', 'Language'])


def scrapper(url):
    if not url.startswith("https://www."):
        url = url.replace('https://', 'https://www.')
    if not url.startswith("http://www."):
        url = url.replace('http://', 'http://www.')
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
            url, 'Website does not exist', 'Website does not exist'
        ]
    else:
        print('Website', url, 'is valid')

        r = requests.get(url)
        text = r.text
        soup = BeautifulSoup(text)

        # Get the title the webpage: title
        if soup.title is None or len(soup.title) == 0:
            title = 'None'

        else:
            title = soup.title.get_text()
            try:
                if detect(title) != 'en':  # check the language first
                    title = translator.translate(title, dest='en').text
            except LangDetectException:
                title = 'None'

        only_words_title = re.sub('[^a-zA-ZäöüÄÖÜß]', ' ', title)
        title_list = only_words_title.split()

        final_list_title = []

        for word in title_list:
            doc = tagger_en.tag_text(word)
            tag = treetaggerwrapper.make_tags(doc)
            for element in tag:
                if element[1] == 'NN' or element[1] == 'NP' or element[1] == 'NNS' or element[1] == 'NPS':
                    if len(element[2].lower()) > 3:
                        final_list_title.append(element[2].lower())
        print("Title", final_list_title)

        # Get the text: text
        text = soup.get_text()
        if text is None or text == 0 or len(text) <= 3:
            text_trans = 'None'
            lang = 'None'
            # df.loc[len(df)] = [
            #     url, 'Bad quality','Bad quality','Bad quality']
        else:
            try:
                lang = detect(text)
                if detect(text) != 'en':  # check the language first
                    text_trans = translator.translate(text, dest='en').text
                else:
                    text_trans = text
            except LangDetectException:
                text_trans = 'None'

        only_words_text = re.sub('[^a-zA-ZäöüÄÖÜß]', ' ', text_trans)
        text_list = only_words_text.split()

        final_list_text = []

        for word in text_list:
            doc = tagger_en.tag_text(word.lower())
            tag = treetaggerwrapper.make_tags(doc)
            for element in tag:
                if element[1] == 'NN' or element[1] == 'NP' or element[1] == 'NNS' or element[1] == 'NPS':
                    final_list_text.append(element[2].lower())
        print("Text", final_list_text)

        # Combining lists
        if len(final_list_text) <= 2:
            final_list_text = []
        if len(final_list_title) <= 2:
            final_list_title = []

        total_list = final_list_text + final_list_title

        total_list_filtered = []
        for word in total_list:
            if word not in stop_words_en and len(word) >= 4:
                total_list_filtered.append(word)

        word_count = {}
        for word in total_list_filtered:
            if word not in word_count:
                word_count[word] = 1
            elif word in word_count:
                word_count[word] += 1
        word_count_f = pd.DataFrame(list(word_count.items()),
                                    columns=['Content', 'Count'])
        word_count_df = word_count_f[word_count_f['Count'] >= 2]

        word_count_df.sort_values("Count", ascending=False, inplace=True)
        word_count_df.reset_index(drop=True, inplace=True)
        # print("The first 10 lines of our most frequent words:", "\n")

        word_count_df['Weight'] = round(word_count_df['Count'] / word_count_df['Count'].sum(), 4)
        # print(word_count_df[word_count_df['Count'] >= 2].head(30))

        # -------------------------------------------------------------------------------
        # Appending the extracted data to final data frame
        # -------------------------------------------------------------------------------

        if len(total_list_filtered) < 5:
            print('Length', len(total_list_filtered))
            df.loc[len(df)] = [
                url, 'Bad quality', 'Bad quality'
            ]
        elif word_count_df.empty:
            df.loc[len(df)] = [
                url, 'Bad quality', 'Bad quality'
            ]
        else:
            # if title == 'None':
            #     df.loc[len(df)] = [
            #         url,
            #         word_count_df.drop(
            #             columns=['Count']).set_index('Content').T.to_dict('records')[0]
            #     ]
            # else:
            df.loc[len(df)] = [
                url, word_count_df.drop(
                    columns=['Count']).set_index('Content').T.to_dict('records')[0], lang
            ]

        df.to_excel(r"C:\Users\Lumitos\Desktop\Table_1234.xlsx")


scrapper('https://www.morinagamilk.co.jp')
