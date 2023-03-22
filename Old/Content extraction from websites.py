import pandas as pd
import requests
import treetaggerwrapper as treetaggerwrapper
from requests.exceptions import ConnectionError, Timeout, ContentDecodingError, InvalidURL, ConnectTimeout, \
    TooManyRedirects, SSLError

from bs4 import BeautifulSoup
from langdetect import detect
from csv import reader
import re
import warnings
from googletrans import Translator
from nltk.corpus import stopwords
from tldextract import extract

warnings.filterwarnings("ignore")

stop_words_en = stopwords.words('english')
stop_words_de = stopwords.words('german')
translator = Translator()
tagger_de = treetaggerwrapper.TreeTagger(TAGLANG='de')
tagger_en = treetaggerwrapper.TreeTagger(TAGLANG='en')
df = pd.DataFrame(columns=['Full_URL', 'Domain', 'Title', 'Content'])


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
            url, 'Website does not exist', 'Website does not exist',
            'Website does not exist'
        ]
    else:
        print('Website', url, 'is valid')

        r = requests.get(url)
        text = r.text
        soup = BeautifulSoup(text)

        # Get the title the webpage: title
        if soup.title is None or len(soup.title) == 0 or '\n' in soup.title:
            title = 'None'
        else:
            title = soup.title.get_text()
            if detect(title) != 'de' and detect(title) != 'en':  # check the language first
                title = translator.translate(title, dest='en').text

            only_words_title = re.sub('[^a-zA-ZäöüÄÖÜß]', ' ', title)
            title_list = only_words_title.split()

            # Print the title of the webpage to the shell
            # title = str(title)
            # print(title)

            final_list_title = []

            if detect(title) == 'de':
                for word in title_list:
                    doc = tagger_de.tag_text(word)
                    tag = treetaggerwrapper.make_tags(doc)
                    for element in tag:
                        if element[1] == 'NN' or element[1] == 'NE':
                            if len(element[2].lower()) > 3:
                                final_list_title.append(element[2].lower())
            else:
                for word in title_list:
                    doc = tagger_en.tag_text(word)
                    tag = treetaggerwrapper.make_tags(doc)
                    for element in tag:
                        if element[1] == 'NN' or element[1] == 'NE':
                            if len(element[2].lower()) > 3:
                                final_list_title.append(element[2].lower())
            # print(final_list_title)

        # Get the text: text
        text = soup.get_text()
        if len(text) <= 3:
            df.loc[len(df)] = [
                url, 'Bad quality', 'Bad quality', 'Bad quality'
            ]
        else:
            if detect(text) != 'de' and detect(text) != 'en':  # check the language first
                text = translator.translate(text, dest='en').text

            # Print text to the shell
            # print(text)

            only_words = re.sub('[^a-zA-ZäöüÄÖÜß]', ' ', text)
            # print(only_words)
            text_list = only_words.split()
            # print(text_list)

            unfiltered_list_text = []
            final_list_text = []

            if detect(text) == 'de':
                for word in text_list:
                    doc = tagger_de.tag_text(word)
                    tag = treetaggerwrapper.make_tags(doc)
                    for element in tag:
                        if element[1] == 'NN' or element[1] == 'NE':
                            unfiltered_list_text.append(element[2].lower())
            else:
                for word in text_list:
                    doc = tagger_de.tag_text(word.lower())
                    tag = treetaggerwrapper.make_tags(doc)
                    for element in tag:
                        if element[1] == 'NN' or element[1] == 'NE':
                            unfiltered_list_text.append(element[2].lower())

            for word in unfiltered_list_text:
                if detect(text) == 'en':
                    if word not in stop_words_en and len(word) >= 4:
                        final_list_text.append(word)
                if detect(text) == 'de':
                    if word not in stop_words_de and len(word) >= 4:
                        final_list_text.append(word)

            # print(final_list_text)

            word_count = {}
            for word in final_list_text:
                if word not in word_count:
                    word_count[word] = 1
                elif word in word_count:
                    word_count[word] += 1
            word_count_df = pd.DataFrame(list(word_count.items()),
                                         columns=['Topic', 'Count'])
            word_count_df.sort_values("Count", ascending=False, inplace=True)
            word_count_df.reset_index(drop=True, inplace=True)
            # print("The first 10 lines of our most frequent words:", "\n")

            word_count_df['Weight'] = round(word_count_df['Count'] / word_count_df['Count'].sum(), 4)
            # print(word_count_df[word_count_df['Count'] >= 2].head(30))

            # ---------------------- Extracting domain from the URL -----------------------

            tsd, td, tsu = extract(url)
            domain = re.sub('[^a-zA-Z0-9]', '', td)

            # -------------------------------------------------------------------------------
            # Appending the extracted data to final data frame
            # -------------------------------------------------------------------------------

            if len(final_list_text) < 5:
                df.loc[len(df)] = [
                    url, 'Bad quality', 'Bad quality', 'Bad quality'
                ]
            else:
                if title == 'None':
                    df.loc[len(df)] = [
                        url, domain, title,
                        word_count_df.drop(
                            columns=['Count']).set_index('Topic').T.to_dict('records')[0]
                    ]
                else:
                    df.loc[len(df)] = [
                        url, domain, ' '.join(final_list_title),
                        word_count_df.drop(
                            columns=['Count']).set_index('Topic').T.to_dict('records')[0]
                    ]

    df.to_excel(r"C:\Users\Lumitos\Desktop\Table.xlsx")


with open(
        r"C:\Users\Lumitos\Desktop\Short_List.csv",
        'r') as read_object:
    csv_file = reader(read_object)
    header = next(csv_file)
    # Check file as empty
    if header is not None:
        # Iterate over each row after the header in the csv
        for row in csv_file:
            # row variable is a list that represents a row in csv
            a = ''.join(row)  # covert it into a string
            scrapper(a)
