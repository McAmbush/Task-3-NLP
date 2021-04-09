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


# Approach

##Data Cleaning

It was observed that some products do not have a primary category. For such products, flipkart webste was scraped and category was obtained. If a category was still not found, the products were removed.
The duplicate categories were merged.
A total of 22 unique categories were identified.
<img src="readme_images/praw%20dataset%20post%20count.png" width="800">

**Body was the primary content missing even in the most popular posts** and subsequent post extractions, 
 hence it was removed as a training feature. 

<img src="readme_images/post%20body%20heatmap.png" width="500">

