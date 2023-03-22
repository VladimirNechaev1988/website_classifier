import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
import warnings
import time
from joblib import dump, load

# Ignore the warnings
warnings.filterwarnings("ignore")

# Create the empty DF for results
df = pd.DataFrame(columns=['Full_URL', 'True Industry', 'TFIDF_SVM'])
acc_tfidf_svm = 0

# ---------------------------------
# ----------Stop words-------------
# ---------------------------------

# Download the standard stopwords
stop_words = stopwords.words('english')

# Get the created stop words list
stop = pd.read_excel(
    r"C:\Users\Lumitos\OneDrive - IU International University of Applied Sciences\IUBH учеба\Thesis\Data "
    r"Sets\STOP_WORDS.xlsx "
)

# Join the two lists and eliminate the duplicates
stop_words = set(stop_words + stop['Words'].tolist())

# Create a word lemmatizer instance for pre-processing the texts
wordnet_lemmatizer = WordNetLemmatizer()

# ---------------------------------
# ---------Get the test set--------
# ---------------------------------

initial_test = pd.read_excel(
    r"C:\Users\Lumitos\OneDrive - IU International University of Applied Sciences\IUBH учеба\Thesis\Data "
    r"Sets\KNN\Final versions\Test_scrapped_with_industry.xlsx ")

# ---------------------------------
# -----Pre-processing the set------
# ---------------------------------

start_time_prep = time.time()
# Convert the column to string and add a white space at the beginning

initial_test['Content'] = initial_test['Content'].astype('str')
initial_test['Content'] = initial_test['Content'].apply(lambda x: ' ' + x)

# Tokenize, lemmatize, lower-case, filter for alphabetical symbols and stop words

test = initial_test[['Full_URL', 'Content', 'Industry']]
test['lists'] = test['Content'].apply(lambda t: [
    wordnet_lemmatizer.lemmatize(t.lower()) for t in word_tokenize(t)
    if t.isalpha() and t.lower() not in stop_words and len(t) >= 3
])
test['texts'] = test['lists'].apply(lambda t: ' '.join(t))
test.drop(columns=['lists', 'Content'], inplace=True)

end_time_prep = time.time()

# ---------------------------------
# ----Load the model from disk-----
# ---------------------------------

svm = load('model.joblib')

# ---------------------------------
# ----Load TF-IDF vectorizer-------
# ---------------------------------

vectorizer = load('vectorizer.joblib')

# ---------------------------------
# --------Test the model-----------
# ---------------------------------

start_time_test = time.time()
# Iterate over each row of the test data and treat it like an X and a y
for index, row in test.iterrows():
    X_test = row[['texts']]
    y_test = row[['Industry']]
    url = row['Full_URL']
    industry = row['Industry']

    # Transform the X_test with TF-IDF vectorizer
    X_test_tfidf = vectorizer.transform(X_test)

    # Use the trained model to make predictions on the testing set
    y_pred_tfidf_svm = svm.predict(X_test_tfidf)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_pred_tfidf_svm, y_test)
    acc_tfidf_svm += accuracy

    # Append the result to the resulting DF
    df.loc[len(df)] = [url, industry, y_pred_tfidf_svm[0]]
    print(url, 'done')

# df.to_excel(r"C:\Users\Lumitos\Desktop\Predictions.xlsx")

print('The accuracy of TFIDF-SVM is:',
      round(((acc_tfidf_svm / len(df)) * 100), 2))
end_time_test = time.time()

elapsed_time_prep = end_time_prep - start_time_prep
print(f"Elapsed time for pre-processing of {len(initial_test)} samples is {round(elapsed_time_prep, 2)} seconds")

elapsed_time_test = end_time_test - start_time_test
print("Elapsed time for testing of", len(df), "samples is", round(elapsed_time_test, 2), "seconds")
