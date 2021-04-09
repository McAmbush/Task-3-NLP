# Flipkart Product Category Prediction
Comparitive analysis of multiple Deep Learning and Machine Learning architectures for predicting the primary category of a given product based on its description and other features.

# Models Used:

1. LSTM
2. CNN
3. BERT
4. Hybrid CNN and LSTM model
5. Random Forest
6. SVM
7. KNN
8. Naive Bayes

# Index 
1.	Project Description
2.	Data Extraction 
3.	Data Pre-processing and Modification
4.	Model development and Testing
5.	Model Summary  
6. References  

# Project Description
 The project aims to develop a pipeline for predicting the primary category of products found on flipkart by using their description and images.
 The Colab notebooks and their description are as follows:

### Colab Notebooks:
*	Data Cleaning and EDA :Extracting relevant data from the given dataset, cleaning it up and visualizing insights about the data.
*	Approach 1: Creating a LSTM based deep learning model and training it on preprocessed data
*	Approach 2: Setting up a BERT model and training it on preprocessed data 
*	Approach 3: Creating a hybrid deep learning model taking both images and descriptions as input and then training it on preprocessed data  
*	Approach 4: Creating 4 different machine learning models and training them on preprocessed data
*	Approach 5: Creating a CNN based deep learning model and training it on preprocessed data
*   Results Analysis: Detailed results of various models

### Libraries Used:
The following libraries were used for data exploration and model development –
* Numpy, Pandas: Data loading and Manipulation
* Sklearn: For Data preprocessing, machine learning models and evaluation metrics
* Seaborn, Matplotlib: Data Visualization
* Requests, BeautifulSoup: Scraping data from flipkart
* Tensorflow: Deep Learning models
* CV2, PIL: Image loading and processing
* TQDM: Timeline View
* Gensim: NLP Preprocessing

# Approach

## Data Cleaning

It was observed that some products do not have a primary category. For such products, flipkart webste was scraped and category was obtained. If a category was still not found, the products were removed.
The duplicate categories were merged.
A total of 22 unique categories were identified. 
Null occurences were removed for descriptions (only 3 were present) and final dataframe was saved.

<img src="Data%20Exp%20Images/Category_count.png" width="600">

#### For the descriptions, the following process was followed:
1. Convert text to lower case
2. Remove punctuations and numbers
3. Remove stopwords
4. Remove contractions
5. Remove stray characters

## Data Visualization

The cleaned data was visualized to gain meaningful insights.
Wordcloud visualization is as follows:

<img src="Data%20Exp%20Images/Wordcloud.png" width="600">

For all categories, most appearing words were plotted. For example, for the clothing category, [women,online,shirt] were most occuring words.

<img src="Data%20Exp%20Images/Clothing_word_freq.png" width="600">

Overall word length distribution stated that the average description had 47 words. For individual categories also, boxplot was plotted for the word distribution.

### Overall Word Length Distribution
<img src="Data%20Exp%20Images/Word_len_distribution.png" width="600">

### Individual Word Length Plot
<img src="Data%20Exp%20Images/Boxplot.png" width="600">

## Model Training and Evaluation

The 8 different models mentioned above were created and trained on the preprocessed data.
For the ML models, TFIDF Vectorizer was used to convert input sentences to vectors.
For the DL models, Tensorflow Tokenizer was used to vectorize the input sentences and the sequences were padding to a max length of 100.

### Best DL Model

Among the deep learning models, BERT performed the best with an f1-score of 97% and macro average of 88%.

<img src="Data%20Exp%20Images/Approach2_val_result.png" width="600">

### Best ML Model

Among the machine learning models, Random Forest performed the best with an f1-score of 97% and macro average of 94% but, the model overfit on training set.

<img src="Data%20Exp%20Images/Approach4_val_rf_result.png" width="600">

### Weights of models:
https://drive.google.com/drive/folders/1DGD2DjOO0-ptIzbtM-DW_hb3LW0Q3vWD?usp=sharing


## Conclusion

The dataset provided has limited datapoints and many urls are outdated. Hence, our models have restriced performance.
Among the deep learning models, BERT performed the best and the loss decayed in just 1 epoch.
Among the machine learning models, random forest yielded higher accuracy than even BERT but, overfitted on training data, which would be an issue as the dataset size grows.

## Future Scope
1. Scrape more data
2. Use Lemmatization or stemming
3. Use Boosted models, like xgboost
4. Use word embeddings like GLoVe
5. Experiment with more complex transformers
6. Optimize the hybrid model with more images per product

## References
1. F1 metrics in keras: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
2. Plotting ROC Curve: analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
3. BERT Tutorial: https://www.tensorflow.org/tutorials/text/classify_text_with_bert
4. https://scikit-learn.org/stable/modules/model_evaluation.html
5. https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
