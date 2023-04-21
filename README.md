# Website Classifier
The project consists of the the following stages:
<li> data collection </li> 
<li> data pre-processing </li>
<li> model selection </li>
<li> hyperparameter tuning </li> 
<li> model deployment as a web service. </li>

## Algorithm Description

The model represents a website classifier that takes a URL as an input and outputs a possible industry that the page belongs to. The following industries are considered in the model
<li>Biotechnology</li>
<li>Drinks</li>
<li>Food </li>
<li>IT</li>
<li>Laboratory analytics and equipment</li>
<li>Logistics </li>
<li>Manufacturing</li>
<li>Recycling</li>  

<br>The algorithm represents a Flask web service that is designed for further integration into a data mining tool, such as Talend Open Studio for processing large amounts of data, hence the simplistic design of the web service and output in JSON format.

## Development of the Model

The development included the following steps:
1. To collect the data for training and testing, the portal curlie.org was used. This is a man-made data pool and search engine which includes a hierarchical directory of websites organized by topic, with each site categorized under a relevant subject. With a search tool on the main page, a desired category was selected, and the output represented a list of web pages from this industry. The URLs of the resulting websites were extracted from the portal. For each industry, 150 websites were selected for use in both training and testing sets. They were organized in a tabular form, resulting in a total of 1200 websites in the collection. 
2. Next, the dataset of URLs was randomly partitioned into training and testing sets with stratification to ensure equal classes proportion. This resulted in 135 URLs per industry for training and 15 for testing.
3. After the data sets were ready, each URL was checked for validity and the textual information was scrapped from the home pages of each website if that was possible. This step is represented in the <b>scrapping_content_curlie</b> script using <b>URL_training</b> and <b>URL_test</b> data sets. The results are saved in <b>training_scrapped_init</b> and <b>test_scrapped_init</b>. 
4. To ensure consistency and quality, features were checked for quality. Those websites that were unavailable, output too little information, or prohibited scrapping, were excluded from the sets. This resulted in 885 URLs for training and 101 for testing. They are saved in <b>training_scrapped_final</b> and <b>test_scrapped_final</b> sets.
5. Data cleaning and modeling can be viewed in the <b>experiments_with_different_models</b> script.

Data preparation consisted of the following steps:
<li>Tokenization</li>
<li>Alpha filtering</li>
<li>Lemmatization</li>
<li>Stop words removal</li>  

<br>Feature engineering included the following techniques:

<li>Bag-of-words</li>
<li>TF-IDF</li>
<li>Word2Vec</li>
<li>GloVe</li>  

<br>Modeling was attempted using the following techniques:

<li>KNN</li>
<li>SVM</li>
<li>Random Forests</li>  

<br>6. The next step is selecting the best-performing model and tuning the parameters. Grid search is found in the <b>GridSearch</b> script where the proper parameters for the TF-IDF SVM algorithm were selected, which ultimately improved the model classification power.
7. Deployment of the model as a web service. Scripts can be found in the app folder.
8. The model was ultimately integrated into Talend Open Studio to work with large data amounts, such as CSV files, etc. The following steps were undertaken:
<br>a) Creating a new job in Talend, inputting the tabular document containing URLs that require classification. 
<br>b) Selecting the column containing URLs from the tabular data by using the mapping component.
<br>c) Iterating over the selected column using the **FlowToIterate** component and extracting each value (URL) at a time using the FixedFlowInput component.
<br>d) With **HttpRequest** component, sending a GET request to the server where the app was deployed and specifying the uniform resource identifier (URI) for inputting the target URL. For that, it is required to complete the app deployment address of the classification route, with a question mark to start the string query and to parse the parameter <i>url</i> with the value extracted from the previous component. 
<br>e) Reading the HTTP response in JSON format by specifying the corresponding value for the accept header.
<br>f) With the component **ExtractJSONFields**, retrieving the value of the input URL, and the respective field of the JSON file, that contains the output industry, by using the JSON query to access the value of the <i>result</i> key. Additionally, one can use the mapping component to form a new output table in a preferable format. 
<br>g) Outputting the resulting data in a desired format, e.g., Excel, CSV.

## Overview of the Folders and Files
### Root
Includes the following files:
<li> <b>experiments_with_different_models</b>: the Python script which creates combinations of different vectorizing techniques with three classification models and outputs a prediction table with the accuracies for each approach.</li>
<li> <b>GridSearch</b>: the Python script which iterates over different hyperparameters of the SVM model and the TF-IDF vectorizer. Uses five-fold cross-validation on the training set. Outputs the hyperparameters with which the model delivers the highest accuracy.</li>
<li> <b>scrapping_content_curlie</b>: the web scrapping script. Uses Beautiful Soup to harvest the textual content from the websites from the URLs in files <b>URL_training</b> and <b>URL_test</b>.</li>
<li> <b>test_model</b>: the Python script that opens the locally saved model and tests its efficiency on the testing set.</li>
<li> <b>TF_IDF_SVM_tuned</b>: the Python script that creates a final SVM model with TF-IDF vectorizer and saves the model and the vectorizer in a joblib format.</li>
<li> <b>requirements_model</b>: the textual service file that includes the list of Python libraries required to be installed in the local environment before proceeding to the scripts.</li>

### Data
The data folder includes 9 Excel files that were used either as primary data or were the output of the pre-processing and modelling algorithms (derived). These are the following data:
<li> <b>URL_training</b>: initial 1080 URLs with assigned industries collected from the Curlie portal for the training set.
<li> <b>URL_test</b>: initial 120 URLs with assigned industries collected from the Curlie portal for the test set.
<li> <b>training_scrapped_init</b>: the initial training set which contains 1080 URLs, their scrapped content, detected language and true industry. This data was scrapped from the websites before filtering for low-quality or inaccessible pages.
<li> <b>training_scrapped_final</b>: the finalized training set which contains 885 URLs, scrapped content, detected language and true industry. This set is derived from the training_scrapped_init after eliminating poor-quality websites.
<li> <b>test_scrapped_init</b>: the initial test set which contains 120 URLs, their scrapped content, detected language and true industry. This data was scrapped from the websites before filtering for low-quality or inaccessible pages.
<li> <b>test_scrapped_final</b>: the finalized test set which contains 101 URLs, scrapped content, detected language and true industry. This set is derived from the test_scrapped_init after eliminating poor-quality websites.
<li> <b>training_set_final</b>: the final pre-processed training set. Contains 8 industries and the content assigned to them. This set is derived from the train-ing_scrapped_final.
<li> <b>predictions</b>: the output of the prediction algorithm on the test set. Contains 101 URLs, their actual industry, and the one output by the classifier.
<li> <b>stop_words</b>: 538 manually selected stop words.

### App
This folder includes **3** downstream folders and a Python file:
#### Folders:
##### Data:
<li> <b>stop_words</b>: 538 manually selected stop words, used for filtering the content of the input website.
<li> <b>model</b>: the trained, fine-tuned, and ready-to-use SVM classification model saved in a joblib file.
<li> <b>vectorizer</b>: TF-IDF vectorizer fitted to the training corpus saved in a joblib file.</li>   

##### Static:
<li> <b>style</b>: the css file that defines the styles to be used by the app interface.</li>

##### Templates:
<li> <b>home_page</b>: the HTML template for the home page of the app which simply represents a header and an input field with a corresponding launch button.
<li> <b>your_industry</b>: the HTML template for the endpoint route of the app.  

##### Script:
<li> <b>app</b>: the main Python script of the web service.
