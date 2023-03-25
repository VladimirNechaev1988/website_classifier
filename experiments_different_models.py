import gensim.downloader as api
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Create a finale table
df = pd.DataFrame(columns=['Full_URL', 'True Industry', 'BOW_KNN', 'BOW_SVM', 'BOW_Forest', 'TFIDF_KNN', 'TFIDF_SVM',
                           'TFIDF_Forest', 'W2V_KNN', 'W2V_SVM', 'W2V_Forest', 'GloVe_KNN', 'GloVe_SVM',
                           'GloVe_Forest'])

# Load the pre-trained Word2Vec model
model = api.load('word2vec-google-news-300')

# Pretrained model for Glove
model_name = 'glove-wiki-gigaword-200'
word_vectors = api.load(model_name)

# Cumulative accuracies
acc_bow_knn = 0
acc_bow_svm = 0
acc_bow_for = 0
acc_tfidf_knn = 0
acc_tfidf_svm = 0
acc_tfidf_for = 0
acc_w2v_knn = 0
acc_w2v_svm = 0
acc_w2v_for = 0
acc_glo_knn = 0
acc_glo_svm = 0
acc_glo_for = 0

# Get the data
initial_training = pd.read_excel(r"data\Training_scrapped_with_industry.xlsx")
initial_test = pd.read_excel(r"data\Test_scrapped_with_industry.xlsx")

initial_training['Content'] = initial_training['Content'].astype('str')
initial_training['Content'] = initial_training['Content'].apply(
    lambda x: ' ' + x)

training = initial_training[['Content', 'Industry']] \
    .groupby(by='Industry').sum()

initial_test['Content'] = initial_test['Content'].astype('str')
initial_test['Content'] = initial_test['Content'].apply(lambda x: ' ' + x)

test = initial_test[['Full_URL', 'Content', 'Industry']]

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

training['lists'] = training['Content'].apply(lambda t: [
    wordnet_lemmatizer.lemmatize(t.lower()) for t in word_tokenize(t)
    if t.isalpha() and t.lower() not in stop_words and len(t) >= 3
])
training['texts'] = training['lists'].apply(lambda t: ' '.join(t))
training.drop(columns=['lists'], inplace=True)
training.reset_index(inplace=True)

test['lists'] = test['Content'].apply(lambda t: [
    wordnet_lemmatizer.lemmatize(t.lower()) for t in word_tokenize(t)
    if t.isalpha() and t.lower() not in stop_words and len(t) >= 3
])
test['texts'] = test['lists'].apply(lambda t: ' '.join(t))
test.drop(columns=['lists'], inplace=True)

X_train = training['texts']
y_train = training['Industry']

for index, row in test.iterrows():
    X_test = row[['texts']]
    y_test = row[['Industry']]
    url = row['Full_URL']
    industry = row['Industry']

    # ---------------------------------
    # -------------BOW-----------------
    # ---------------------------------

    # Create a count vectorizer and fit it to the training and test data
    count_vect = CountVectorizer()
    X_train_bow = count_vect.fit_transform(X_train)
    X_test_bow = count_vect.transform(X_test)

    # -------------KNN-BOW-----------------

    # Train the KNN model using the TF-IDF features and labels
    knn_bow = KNeighborsClassifier(n_neighbors=2, algorithm='brute')
    knn_bow.fit(X_train_bow, y_train)

    # Use the trained model to make predictions on the testing set
    y_pred_bow_knn = knn_bow.predict(X_test_bow)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred_bow_knn)
    acc_bow_knn += accuracy

    # -------------SVM-BOW-----------------

    # Train the SVM model using the TF-IDF features and labels
    svm = SVC()
    svm.fit(X_train_bow, y_train)

    # Use the trained model to make predictions on the testing set
    y_pred_bow_svm = svm.predict(X_test_bow)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred_bow_svm)
    acc_bow_svm += accuracy

    # -------------Forest-BOW-----------------

    # Train random forest classifier
    n_estimators = 100  # number of trees in the forest
    max_depth = 10  # maximum depth of the tree
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train_bow, y_train)

    # Use the trained model to make predictions on the testing set
    y_pred_bow_forest = clf.predict(X_test_bow)

    # Evaluate the random forest classifier on the test data
    accuracy = clf.score(X_test_bow, y_test)
    acc_bow_for += accuracy

    # ---------------------------------
    # -------------TF-IDF--------------
    # ---------------------------------

    # Create a TF-IDF vectorizer and fit it to the training and test data
    vectorizer = TfidfVectorizer(min_df=0.2,
                                 max_df=5,
                                 sublinear_tf=True,
                                 use_idf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # -------------KNN-TF-IDF-----------------

    # Train the KNN model using the TF-IDF features and labels
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train_tfidf, y_train)

    # Use the trained model to make predictions on the testing set
    y_pred_tfidf_knn = knn.predict(X_test_tfidf)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_pred_tfidf_knn, y_test)
    acc_tfidf_knn += accuracy

    # -------------SVM-TF-IDF-----------------

    # Train the SVM model using the TF-IDF features and labels
    svm = SVC()
    svm.fit(X_train_tfidf, y_train)

    # Use the trained model to make predictions on the testing set
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred_tfidf_svm = svm.predict(X_test_tfidf)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_pred_tfidf_svm, y_test)
    acc_tfidf_svm += accuracy

    # -------------Forest-TF-IDF-----------------

    # Train random forest classifier
    n_estimators = 100  # number of trees in the forest
    max_depth = 10  # maximum depth of the tree
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train_tfidf, y_train)

    # Use the trained model to make predictions on the testing set
    y_pred_tfidf_forest = clf.predict(X_test_tfidf)

    # Evaluate the random forest classifier on the test data
    accuracy = clf.score(X_test_tfidf, y_test)
    acc_tfidf_for += accuracy

    # ---------------------------------
    # -----------Word2Vec--------------
    # ---------------------------------

    # Define a function to generate embeddings for each document

    def generate_doc_embedding(doc):
        words = doc.split()
        embeddings = []
        for w in words:
            try:
                embeddings.append(model[w])
            except KeyError:
                # Ignore words that are not in the vocabulary
                pass
        # Take the mean of all embeddings to generate a single document embedding
        if len(embeddings) > 0:
            return np.mean(embeddings, axis=0)
        else:
            # Return a zero vector if no embeddings were found
            return np.zeros(model.vector_size)


    # Generate embeddings for each document in the training set
    X_train_embeddings = [generate_doc_embedding(doc) for doc in X_train]

    # Generate embeddings for each document in the testing set
    X_test_embeddings = [generate_doc_embedding(doc) for doc in X_test]

    # -------------KNN-Word2Vec-----------------

    # Train the KNN model using the training set embeddings and labels
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train_embeddings, y_train)

    # Use the trained model to make predictions on the testing set embeddings
    y_pred_word2vec_knn = knn.predict(X_test_embeddings)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_pred_word2vec_knn, y_test)
    acc_w2v_knn += accuracy

    # -------------SVM-Word2Vec-----------------

    # Train the SVM model using the TF-IDF features and labels
    svm = SVC()
    svm.fit(X_train_embeddings, y_train)

    # Use the trained model to make predictions on the testing set embeddings
    y_pred_word2vec_svm = svm.predict(X_test_embeddings)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_pred_word2vec_svm, y_test)
    acc_w2v_svm += accuracy

    # -------------Forest-Word2Vec-----------------

    # Train random forest classifier
    n_estimators = 100  # number of trees in the forest
    max_depth = 10  # maximum depth of the tree
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train_embeddings, y_train)

    # Use the trained model to make predictions on the testing set
    y_pred_word2vec_forest = clf.predict(X_test_embeddings)

    # Evaluate the random forest classifier on the test data
    accuracy = clf.score(X_test_embeddings, y_test)
    acc_w2v_for += accuracy

    # ---------------------------------
    # -----------GloVe--------------
    # ---------------------------------

    # Load data
    labels_training = training['Industry'].values
    texts_training = training['texts'].values

    labels_test = y_test.values
    texts_test = X_test.values

    # Convert input text to GloVe embeddings
    embed_dim = 200  # dimensionality of the GloVe embeddings
    max_seq_len = 50  # maximum length of input sequence
    X_train_glove = np.zeros((len(texts_training), max_seq_len, embed_dim))
    for i, text in enumerate(texts_training):
        for j, word in enumerate(text.split()):
            if j >= max_seq_len:
                break
            if word in word_vectors.key_to_index:
                X_train_glove[i, j, :] = word_vectors.get_vector(word)

    X_test_glove = np.zeros((len(texts_test), max_seq_len, embed_dim))
    for i, text in enumerate(texts_test):
        for j, word in enumerate(text.split()):
            if j >= max_seq_len:
                break
            if word in word_vectors.key_to_index:
                X_test_glove[i, j, :] = word_vectors.get_vector(word)

    # Split data into training and testing sets

    y_train_glove = labels_training
    y_test_glove = labels_test

    # Flatten input data for SVM
    X_train_flattened = X_train_glove.reshape(X_train_glove.shape[0], -1)
    X_test_flattened = X_test_glove.reshape(X_test_glove.shape[0], -1)

    # -------------KNN-GloVe-----------------

    # Train the KNN model using the training set embeddings and labels
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train_flattened, y_train_glove)

    # Use the trained model to make predictions on the testing set embeddings
    y_pred_glove_knn = knn.predict(X_test_flattened)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test_glove, y_pred_glove_knn)
    acc_glo_knn += accuracy

    # -------------SVM-GloVe-----------------

    # Train SVM on the flattened GloVe embeddings
    clf = SVC()
    clf.fit(X_train_flattened, y_train_glove)

    # Predict labels for the test data
    y_pred_glove_svm = clf.predict(X_test_flattened)

    # Evaluate the SVM classifier on the test data
    accuracy = accuracy_score(y_test_glove, y_pred_glove_knn)
    acc_glo_svm += accuracy

    # -------------Forest-GloVe-----------------

    # Train random forest classifier
    n_estimators = 100  # number of trees in the forest
    max_depth = 10  # maximum depth of the tree
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train_flattened, y_train_glove)

    # Use the trained model to make predictions on the testing set
    y_pred_glove_forest = clf.predict(X_test_flattened)

    # Evaluate the random forest classifier on the test data
    accuracy = accuracy_score(y_test_glove, y_pred_glove_knn)
    acc_glo_for += accuracy

    df.loc[len(df)] = [
        url, industry, y_pred_bow_knn[0], y_pred_bow_svm[0], y_pred_bow_forest[0], y_pred_tfidf_knn[0],
        y_pred_tfidf_svm[0],
        y_pred_tfidf_forest[0], y_pred_word2vec_knn[0], y_pred_word2vec_svm[0], y_pred_word2vec_forest[0],
        y_pred_glove_knn[0], y_pred_glove_svm[0], y_pred_glove_forest[0]
    ]
    print(url, 'done')

# df.to_excel(r"data\predictions_all_models.xlsx")
print('The accuracy of BOW-KNN is:', round((acc_bow_knn / len(df)) * 100, 2))
print('The accuracy of BOW-SVM is:', round((acc_bow_svm / len(df)) * 100, 2))
print('The accuracy of BOW-Forest is:', round((acc_bow_for / len(df)) * 100, 2))
print('The accuracy of TFIDF-KNN is:', round((acc_tfidf_knn / len(df)) * 100, 2))
print('The accuracy of TFIDF-SVM is:', round((acc_tfidf_svm / len(df)) * 100, 2))
print('The accuracy of TFIDF-Forest is:', round((acc_tfidf_for / len(df)) * 100, 2))
print('The accuracy of W2V-KNN is:', round((acc_w2v_knn / len(df)) * 100, 2))
print('The accuracy of W2V-SVM is:', round((acc_w2v_svm / len(df)) * 100, 2))
print('The accuracy of W2V-Forest is:', round((acc_w2v_for / len(df)) * 100, 2))
print('The accuracy of GloVe-KNN is:', round((acc_glo_knn / len(df)) * 100, 2))
print('The accuracy of GloVe-SVM is:', round((acc_glo_svm / len(df)) * 100, 2))
print('The accuracy of GloVe-Forest is:', round((acc_glo_for / len(df)) * 100, 2))
