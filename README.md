**Sentiment Analysis of Amazon Office Product Reviews**

**Project Overview**
This project performs sentiment analysis on Amazon office product reviews using various machine learning techniques. The goal is to classify reviews as positive or negative based on their content.


**Table of Contents**
Dataset Preparation
Data Cleaning
Pre-processing
TF-IDF Feature Extraction
Machine Learning Models
Results
Requirements
Usage

**Dataset Preparation**
-The dataset is sourced from Amazon reviews for office products.
-We use pandas to read the TSV file from a URL and load it into a DataFrame.
-Only the 'review_body' and 'star_rating' columns are retained for analysis.

**Data Cleaning**
The following cleaning steps are performed:
Convert text to lowercase
Remove HTML tags and URLs
Expand contractions
Remove non-alphabetic characters and special symbols
Remove extra spaces

**Pre-processing**
Text pre-processing includes:
Removing stop words
Lemmatization to reduce words to their base form
TF-IDF Feature Extraction
TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text data into numerical features.
8000 features are extracted using scikit-learn's TfidfVectorizer.

**Machine Learning Models**
Four different models are implemented and compared:
**Perceptron**
**Support Vector Machine (SVM)**
**Logistic Regression**
**Naive Bayes**


**Results**

Performance metrics (Accuracy, Precision, Recall, F1 Score) are calculated for each model on both training and testing datasets.

**Requirements**
Python 3.x
pandas
numpy
nltk
scikit-learn
BeautifulSoup
matplotlib


**Clone the repository:**
git clone https://github.com/sriramgurazada/Sentimental-Analysis.git


**Run the script:**
python sentimentalAnalysis.py


