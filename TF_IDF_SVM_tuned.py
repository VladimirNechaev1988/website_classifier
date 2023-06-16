import time
import warnings

import pandas as pd
from joblib import dump
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Ignore the warnings
warnings.filterwarnings("ignore")

# ---------------------------------
# ----------Stop words-------------
# ---------------------------------

# Download the standard stopwords
stop_words = stopwords.words('english')

# Get the created stop words list
stop = pd.read_excel(r"data\stop_words.xlsx")

# Join the two lists and eliminate the duplicates
stop_words = set(stop_words + stop['Words'].tolist())

# Create a word lemmatizer instance for pre-processing the texts
wordnet_lemmatizer = WordNetLemmatizer()

# ---------------------------------
# ------Get the training set-------
# ---------------------------------

initial_training = pd.read_excel(r"data\training_scrapped_final.xlsx")

# ---------------------------------
# -----Pre-processing the sets-----
# ---------------------------------

start_time_prep = time.time()

# Convert the column to string and add a white space at the beginning
initial_training['Content'] = initial_training['Content'].astype('str')
initial_training['Content'] = initial_training['Content'].apply(
    lambda x: ' ' + x)

# Group by the industries and concatenate the 'Content' columns
training = initial_training[['Content', 'Industry']] \
    .groupby(by='Industry').sum()

# Tokenize, lemmatize, lower-case, filter for alphabetical symbols and stop words
training['lists'] = training['Content'].apply(lambda t: [
    wordnet_lemmatizer.lemmatize(t.lower()) for t in word_tokenize(t)
    if t.isalpha() and t.lower() not in stop_words and len(t) >= 3
])
training['texts'] = training['lists'].apply(lambda t: ' '.join(t))
training.drop(columns=['lists', 'Content'], inplace=True)
training.reset_index(inplace=True)

# ---------------------------------
# Define X and y in training set---
# ---------------------------------

X_train = training['texts']
y_train = training['Industry']

# ---------------------------------
# ----Create TF-IDF vectorizer-----
# ---------------------------------

vectorizer = TfidfVectorizer(min_df=0.05,
                             max_df=0.5,
                             sublinear_tf=True,
                             use_idf=True)

# ---------------------------------
# ----Create the SVM model---------
# ---------------------------------

svm = SVC(C=1, kernel='rbf')

# Fit and transform the X_train data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train the SVM model using the TF-IDF features and labels
svm.fit(X_train_tfidf, y_train)

# Save the model & vectorizer
dump(svm, r"model\model.joblib")
dump(vectorizer, r"model\vectorizer.joblib")

end_time_prep = time.time()

# Print the time required for the pre-processing
elapsed_time_prep = end_time_prep - start_time_prep
print(
    f"Elapsed time for pre-processing of {len(initial_training)} samples: {round(elapsed_time_prep, 2)} seconds")
