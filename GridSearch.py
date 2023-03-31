import time
import warnings

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Ignore the warnings
warnings.filterwarnings("ignore")

# ---------------------------------
# ----------Stop words-------------
# ---------------------------------

# Download the standard stopwords
stop_words = stopwords.words('english')

# Get the created stop words list
stop = pd.read_excel(r"data\STOP_WORDS.xlsx")

# Join the two lists and eliminate the duplicates
stop_words = set(stop_words + stop['Words'].tolist())

# Create a word lemmatizer instance for pre-processing the texts
wordnet_lemmatizer = WordNetLemmatizer()

# ---------------------------------
# ------Get the training set-------
# ---------------------------------

train_grid = pd.read_excel(r"data\training_scrapped_final.xlsx ")

# ---------------------------------
# ---------Hyperparams-------------
# ---------------------------------

# Create a grid search training set and pre-process it in the same manner
train_grid['lists'] = train_grid['Content'].apply(lambda t: [
    wordnet_lemmatizer.lemmatize(t.lower()) for t in word_tokenize(t)
    if t.isalpha() and t.lower() not in stop_words and len(t) >= 3
])
train_grid['texts'] = train_grid['lists'].apply(lambda t: ' '.join(t))
train_grid.drop(columns=['lists', 'Content'], inplace=True)

start_time_grid = time.time()
# Create a pipeline with the TF-IDF vectorizer and SVM model
pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('svm', SVC())])

# Define the parameter grid for the grid search
parameters = {
    'tfidf__min_df': [0.1, .05, .5],
    'tfidf__max_df': [0.5, 1.0],
    'tfidf__sublinear_tf': [True, False],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'svm__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    'svm__C': [1, 5, 10]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline,
                           parameters,
                           cv=5,
                           n_jobs=-1,
                           verbose=1,
                           scoring='accuracy')
grid_search.fit(train_grid['texts'], train_grid['Industry'])

# Print the best parameters and the corresponding accuracy score
print("Best parameters: ", grid_search.best_params_)
print("Best accuracy: ", grid_search.best_score_)
end_time_grid = time.time()
elapsed_time_grid = end_time_grid - start_time_grid
print(f"Elapsed time for testing: {elapsed_time_grid} seconds")
