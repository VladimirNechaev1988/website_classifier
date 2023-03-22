import re

import requests
import pandas as pd
import treetaggerwrapper
from bs4 import BeautifulSoup
import warnings
from csv import reader
from googletrans import Translator

from langdetect import detect
from requests.exceptions import ConnectionError, Timeout, ContentDecodingError, InvalidURL, ConnectTimeout, \
    TooManyRedirects
from urllib import request
from urllib.error import URLError
from socket import timeout
import string
import translators as ts

# enable ignoring warnings
warnings.filterwarnings("ignore")

# --------------------------------------
# create empty variables that will be used in the functions and main body
# --------------------------------------

big_df = pd.DataFrame(
    columns=["Term", "Industry_1", "Industry_2", "Industry_3", "Industry_Mode"])  # main data frame with the overview
valid = ''  # variable for checking if the website exists
redirected_url = ''  # variable for getting a redirecting URL

tagger_de = treetaggerwrapper.TreeTagger(TAGLANG='de')
tagger_en = treetaggerwrapper.TreeTagger(TAGLANG='en')
translator = Translator()

lookup = pd.read_csv(
    r"C:\Users\Lumitos\OneDrive - IU International University of Applied Sciences\IUBH учеба\Thesis\Data Sets\lookup.csv",
    encoding='unicode_escape')


# --------------------------------------
# define a function that checks if a URL is valid
# --------------------------------------


def most_frequent(li):
    return max(set(li), key=li.count)


def valid_url(input_url):
    if (not (input_url.startswith('http://'))) and (not (input_url.startswith('https://'))):
        input_url = 'http://' + input_url

    try:
        requests.get(input_url, timeout=10, headers={'Connection': 'close'})  # specify timeout
        global valid
        valid = 'yes'  # assign the value for the global variable

    except (ConnectionError, Timeout, ContentDecodingError, InvalidURL, ConnectTimeout, TooManyRedirects,
            ConnectionResetError):  # no connection error (includes SSL error)
        valid = 'no'


# --------------------------------------
# define a function that redirects the URL if it is not valid
# --------------------------------------

def redir_url(input_url):
    try:
        response = request.urlopen(input_url, timeout=5)
        global redirected_url
        redirected_url = response.geturl()
        global valid
        valid = 'yes'

    except (URLError, timeout, ConnectionResetError):
        valid = 'no'
        redirected_url = ''


# --------------------------------------
# define the main function that will perform the topicizing and save results locally
# --------------------------------------

def classifier(url):
    # --------------------------------------
    # if the URL doesn't begin with HTTP, append it
    # --------------------------------------

    if (not (url.startswith('http://'))) and (not (url.startswith('https://'))):
        url = 'http://' + url

    # --------------------------------------
    # Check if the URL is valid
    # --------------------------------------

    valid_url(url)

    # --------------------------------------
    # If the URL is not valid, check if it redirects
    # --------------------------------------
    if valid == 'no':

        redir_url(url)
        if len(redirected_url) != 0:
            # url = redirected_url
            print('Website', url, 'was redirected')
            # big_df.loc[len(big_df)] = [url, 'Redirected']
        else:
            big_df.loc[len(big_df)] = [url, 'Website does not exist', 'Website does not exist',
                                       'Website does not exist', 'Website does not exist']
            print('Website', url, 'does not exist')
            return
    # --------------------------------------
    # send a BS request and retrieve the data
    # --------------------------------------

    r = requests.get(url)
    text = r.text
    soup = BeautifulSoup(text)

    # Get the text: text
    text = soup.get_text()

    # --------------------------------------
    # translate the text
    # --------------------------------------

    if detect(text) != 'en':  # check the language first
        # text = ts.translate_text(text, to_language='en')
        text = translator.translate(text, dest='en').text

    # --------------------------------------
    # leave only alpha characters and split
    # --------------------------------------

    only_words = re.sub('[^a-zA-ZäöüÄÖÜß]', ' ', text)
    # print(only_words)
    text_list = only_words.split()

    # --------------------------------------
    # filter out nouns
    # --------------------------------------

    unfiltered_list_text = []
    final_list_text = []

    for word in text_list:
        doc = tagger_de.tag_text(word.lower())
        tag = treetaggerwrapper.make_tags(doc)
        for element in tag:
            if element[1] == 'NN' or element[1] == 'NE' or element[1] == 'NNS' or element[1] == 'FM' or element[1] \
                    == 'ADJD':
                unfiltered_list_text.append(element[2].lower())
            else:
                unfiltered_list_text.append('')

    # --------------------------------------
    # final list will contain words longer than 4 chars
    # --------------------------------------
    for word in unfiltered_list_text:
        if len(word) >= 4:
            final_list_text.append(word)

    # --------------------------------------
    # if the list is long enough, perform join
    # --------------------------------------

    if len(final_list_text) > 5:
        # convert list onto frame
        frame = pd.DataFrame(final_list_text, columns=['Content'])
        # perform a join
        merged = frame.merge(lookup, how='left', on='Content').set_index('Content')
        merged = merged.groupby(['Content', 'Industry']).max().reset_index()
        merged.set_index('Content')
        # identify highest probability industries
        industry_list = []
        for content in merged.index.unique():
            if len(merged.loc[[content]]) != 1:
                industry_list.append(merged.loc[content][merged.loc[content]['Probability'] ==
                                                         merged.loc[content]['Probability'].max()]
                                     [['Industry']].squeeze())
            else:
                industry_list.append(merged.loc[content]['Industry'])
        print('For the website', url, 'the industry is:', most_frequent(industry_list))
    # else:
    #     big_df.loc[len(big_df)] = [
    #         url, 'Bad quality', 'Bad quality', 'Bad quality', 'Bad quality']


classifier('http://www.tuvit.de/')
