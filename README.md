# Website Classifier
The project consists of the following stages:
*  data collection    
*  data pre-processing   
*  model selection   
*  hyperparameter tuning    
*  model deployment as a web service.   

## Algorithm Description

The model represents a website classifier that takes a URL as an input and outputs a possible industry that the page might belong to. The following industries are considered in the model:  
* Biotechnology  
* Drinks  
* Food   
* IT  
* Laboratory analytics and equipment  
* Logistics   
* Manufacturing  
* Recycling      

The algorithm represents a Flask web service that is designed for further integration into a data mining tool, such as *Talend Open Studio* for processing large amounts of data, hence the simplistic design of the web service and output in JSON format.

## Workflow Explanation

The development included the following steps:
1. **Data collection**. To collect the data for training and testing, the portal *curlie.org* was used. This is a man-made data pool and search engine which includes a hierarchical directory of websites organized by topic, with each site categorized under a relevant subject. With a search tool on the main page, a desired category was selected, and the output represented a list of web pages from this industry. The URLs of the resulting websites were extracted from the portal. For each industry, 150 websites were selected for use in both training and testing sets. They were organized in a tabular form, resulting in a total of 1200 websites in the collection. 
2. **Training and test**. Next, the dataset of URLs was randomly partitioned into training and testing sets with stratification to ensure equal classes proportion. This resulted in 135 URLs per industry for training and 15 for testing.
3. **Scrapping**. After the data sets were ready, each URL was checked for validity and the textual information was scrapped from the home pages of each website if that was possible. This step is represented in the **scrapping_content_curlie** script using **URL_training** and **URL_test** data sets. The results are saved in **training_scrapped_init** and **test_scrapped_init**. 
4. **Quality check**. To ensure consistency and quality, scrapped texts were checked for quality. Those websites that were unavailable, output too little information, or prohibited scrapping, were excluded from the sets. This resulted in 885 URLs for training and 101 for testing. They are saved in **training_scrapped_final** and **test_scrapped_final** sets.
5. **Data preparation and modeling**. Data cleaning and preparation can be viewed in the **experiments_with_different_models** script.

    **Data preparation** consisted of the following steps:
    * Tokenization
    * Alpha filtering
    * Lemmatization
    * Stop words removal 

    **Feature engineering** included the following techniques:
    
    * Bag-of-words  
    * TF-IDF
    * Word2Vec
    * GloVe  
    
    **Modeling** was attempted using the following techniques:
    
    * KNN
    * SVM
    * Random Forests 

6. **Model selection**. After the experiments, the best performing model was chosen. The accuracy of the model was assessed by its ability to correctly assign an industry, which was either correct or incorrect. The cumulative number of correct predictions was divided by the total number of test samples to deliver the accuracy on the scale of 0 to 100. 
7. **Model tuning**. Grid search is found in the **GridSearch** script where the proper parameters for the TF-IDF SVM algorithm were selected, which ultimately improved the model classification power.  
8. **Deployment of the model** Ultimately the model was deployed as a simple web service. Scripts can be found in the **app** folder.  
9. **Integration**. The model was ultimately integrated into Talend Open Studio to work with large data amounts, such as CSV files, etc. The following steps were undertaken:    
a) Creating a new job in Talend, inputting the tabular document containing URLs that require classification.   
b) Selecting the column containing URLs from the tabular data by using the mapping component.  
c) Iterating over the selected column using the **FlowToIterate** component and extracting each value (URL) at a time using the FixedFlowInput component.  
d) With **HttpRequest** component, sending a GET request to the server where the app was deployed and specifying the uniform resource identifier (URI) for inputting the target URL. For that, it is required to complete the app deployment address of the classification route, with a question mark to start the string query and to parse the parameter *url* with the value extracted from the previous component.   
e) Reading the HTTP response in JSON format by specifying the corresponding value for the accept header.  
f) With the component **ExtractJSONFields**, retrieving the value of the input URL, and the respective field of the JSON file, that contains the output industry, by using the JSON query to access the value of the *result* key. Additionally, one can use the mapping component to form a new output table in a preferable format.   
g) Outputting the resulting data in a desired format, e.g., Excel, CSV.  

## Overview of the Folders and Files
### Root
Includes the following files:  
* **experiments_with_different_models**: the Python script which creates combinations of different vectorizing techniques with three classification models and outputs a prediction table with the accuracies for each approach.  
* **GridSearch**: the Python script which iterates over different hyperparameters of the SVM model and the TF-IDF vectorizer. Uses five-fold cross-validation on the training set. Outputs the hyperparameters with which the model delivers the highest accuracy.  
* **scrapping_content_curlie**: the web scrapping script. Uses Beautiful Soup to harvest the textual content from the websites from the URLs in files **URL_training** and **URL_test**.  
* **test_model**: the Python script that opens the locally saved model and tests its efficiency on the testing set.  
* **TF_IDF_SVM_tuned**: the Python script that creates a final SVM model with TF-IDF vectorizer and saves the model and the vectorizer in a *joblib* format.  
* **requirements_model**: the textual service file that includes the list of Python libraries required to be installed in the local environment before proceeding to the scripts by running the following command in the terminal: **pip install -r requirements.txt**.  

### Data
The data folder includes 9 Excel files that were used either as primary data or were the output of the pre-processing and modelling algorithms (derived). These are the following data:  
* **URL_training**: initial 1080 URLs with assigned industries collected from the Curlie portal for the training set.  
* **URL_test**: initial 120 URLs with assigned industries collected from the Curlie portal for the test set.  
* **training_scrapped_init**: the initial training set which contains 1080 URLs, their scrapped content, detected language and true industry. This data was scrapped from the websites before filtering for low-quality or inaccessible pages.  
* **training_scrapped_final**: the finalized training set which contains 885 URLs, scrapped content, detected language and true industry. This set is derived from the training_scrapped_init after eliminating poor-quality websites.  
* **test_scrapped_init**: the initial test set which contains 120 URLs, their scrapped content, detected language and true industry. This data was scrapped from the websites before filtering for low-quality or inaccessible pages.  
* **test_scrapped_final**: the finalized test set which contains 101 URLs, scrapped content, detected language and true industry. This set is derived from the test_scrapped_init after eliminating poor-quality websites.  
* **training_set_final**: the final pre-processed training set. Contains 8 industries and the content assigned to them. This set is derived from the train-ing_scrapped_final.  
* **predictions**: the output of the prediction algorithm on the test set. Contains 101 URLs, their actual industry, and the one output by the classifier.  
* **stop_words**: 538 manually selected stop words.  

### App
This folder includes **3** downstream folders and a Python file:
#### Folders:
##### Data:  
* **stop_words**: 538 manually selected stop words, used for filtering the content of the input website.  
* **model**: the trained, fine-tuned, and ready-to-use SVM classification model saved in a joblib file.  
* **vectorizer**: TF-IDF vectorizer fitted to the training corpus saved in a joblib file.   

##### Static:
* **style**: the css file that defines the styles to be used by the app interface.  

##### Templates:
* **home_page**: the HTML template for the home page of the app which simply represents a header and an input field with a corresponding launch button.  
* **your_industry**: the HTML template for the endpoint route of the app.    

##### Script:
* **app**: the main Python script of the web service.