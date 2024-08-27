
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
 


# - Importing and installing all the required packages 
# 

# In[2]:


get_ipython().system(" pip install bs4 # in case you don't have it installed")

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# ## 1. Dataset Preparation

# ###  Read Data

# - Here we are using pandas library to read a tsv file from url having amazon reviews for office products. 
# Next we are loading data into data frame and i am displaying first 10 lines of the dataframe.

# In[3]:


url = "https://web.archive.org/web/20201127142707if_/https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Office_Products_v1_00.tsv.gz"
df=pd.read_csv(url,compression="gzip",header=0,sep='\t',on_bad_lines='skip')
df.head(10)


# In[4]:


df_main=df.copy()

df.head(10)


# 
# ### Reviews and Ratings

# Here in the below code i am dropping off all the null values from dataframe and picking only 2 columns which are needed i.e "review_body" and "star_ratings"

# In[5]:


# Dropping the Null Values
df=df[['review_body','star_rating']].dropna()
df.head(10)


# In[6]:


df=df.groupby('star_rating').filter(lambda x : len(x)>3)


# Here below we are making sure that the ratings all are in integer format so that in the next steps it will be easy to perform actions on the dataframe.

# In[7]:


df = df.astype({"star_rating": int},errors='ignore')


# Here as asked in the question we are displaying 3 sample reviews 

# In[8]:


# Displaying 
sample_reviews = df.sample(3)
print("3 Reviews:")
print(sample_reviews)
df.head(10)


# Here below We are counting the number of individual rating counts and displaying the count values, Just for visualisation of counts/occurences i have made a small bar graph.

# In[9]:


import matplotlib.pyplot as plt
star_ratings_counts = df['star_rating'].value_counts()

# Print the count for each rating
for rating, count in star_ratings_counts.items():
    print(f"StarRating {int(rating)}: {count} occurrences")

# Convert the index and values to lists for plotting
ratings = star_ratings_counts.index.astype(int).tolist()
counts = star_ratings_counts.tolist()
#plotting the bar graph
plt.bar(ratings, counts, color='skyblue')
plt.xlabel('star_rating')
plt.ylabel('Occurrences')
plt.title('Distribution of Star Ratings')
plt.show()


# Here below we are creating a new column called class and classifying the ratings into 3 categories but we are just considering only 2 positive and negative review ratings, the ratings below 3 are given the value 0 and the ratings above 3 are given the value 1. We are ignoring the values with rating 3.

# In[10]:


# Creating a new column and classifing the reviews into 3 categories.
def label_race (row):
    if row['star_rating'] == 1 :
        return 0
    if row['star_rating'] == 2 :
        return 0
    if row['star_rating'] == 4:
        return 1
    if row['star_rating'] == 5:
        return 1
df['class'] = df.apply (lambda row: label_race(row), axis=1)
print(df)


# In[11]:


class_counts = df['class'].value_counts()

# Print the count for each rating
for rating, count in  class_counts.items():
    print(f"class {int(rating)}: {count} occurrences")


# Here we are checking if there are any null values and dropping off from the dataframe. Just to make sure the null values are not present in dataframe.

# In[12]:


df=df.dropna()
print(df)


# Below we are checking for the class with value 1 and taking 100000 samples and same with the class 0, marking them as positive and negative reviews and concating into 1 dataframe again calling it as df1.

# In[13]:


positive_reviews = df[df['class'] == 1].sample(min(100000, len(df[df['class'] == 1])), random_state=42, replace=False)
negative_reviews = df[df['class'] == 0].sample(min(100000, len(df[df['class'] == 0])), random_state=42, replace=False)
df1 = pd.concat([positive_reviews, negative_reviews])
df1.head(10)


# In[14]:


class_counts = df1['class'].value_counts()

# Print the count for each rating
for rating, count in  class_counts.items():
    print(f"class {int(rating)}: {count} occurrences")


# Below we are calculating the average length of strings in the 'review_body' column of the DataFrame df1 before any data cleaning or processing and prints the result.

# In[15]:


# Calculating the length of strings in review body and taking their mean.
len_before_cleaning=df1['review_body'].str.len().mean()
print(len_before_cleaning)


# ## 2. Data Cleaning

# - Here below we are making sure our dataframe is perfect to use and for pre processing so we are performing few actions.
# - First of all we are converting every string into lowercase.
# - Removing html and other tags from the dataframe to make sure the dataframe is clean.
# - Expanding the contraction words, this is an important step as in the data we can notice many short forms, and here in this step we are making sure the words are expanded.
# - Next we are making sure we delete all the special characters from reviews, as we cant process the special characters.
# - We are removing numbers from the reviews and making sure there are no extra spaces.

# In[16]:


#in data cleaning 
#first we are converting every string into lower case.

df2=df1.copy()
df2['review_body'] = df2['review_body'].str.lower()
df2.head(10)


# In[17]:


# Below is the function to remove HTML tags
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Below we are Applying HTML tag removal to 'review_body'
df2['review_body'] = df2['review_body'].apply(remove_html_tags)

# Below we are Removeing "www." from 'review_body'
df2['review_body'] = df2['review_body'].str.replace('www\.', '', regex=True)

# Here we are rinting the updated DataFrame
print(df2['review_body'])


# In[18]:


# removing the HTML and URLs from the reviews
#df2['review_body'] = df2['review_body'].str.replace(r'<[^<>]*>', '', regex=True)
#df2.head(10)


# In[19]:


# Example contraction mapping dictionary
contraction_mapping = {
    "ain't": "am not", "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would / he had",
    "he'll": "he will / he shall",
    "he's": "he is / he has",
    "I'd": "I would / I had",
    "I'll": "I will / I shall",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would / it had",
    "it'll": "it will / it shall",
    "it's": "it is / it has",
    "let's": "let us",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would / she had",
    "she'll": "she will / she shall",
    "she's": "she is / she has",
    "shouldn't": "should not",
    "that's": "that is / that has",
    "there's": "there is / there has",
    "they'd": "they would / they had",
    "they'll": "they will / they shall",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would / we had",
    "we'll": "we will / we shall",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will / what shall",
    "what're": "what are",
    "what's": "what is / what has",
    "what've": "what have",
    "where's": "where is / where has",
    "who'd": "who would / who had",
    "who'll": "who will / who shall",
    "who're": "who are",
    "who's": "who is / who has",
    "who've": "who have",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would / you had",
    "you'll": "you will / you shall",
    "you're": "you are",
    "you've": "you have",
    
}


# In[20]:


# Function to expand contractions
def expand_contractions(text, contraction_mapping):
    for contraction, expansion in contraction_mapping.items():
        text = text.replace(contraction, expansion)
    return text

df2['review_body'] = expand_contractions(df2['review_body'], contraction_mapping)

df2.head(10)


# In[21]:


#removing non alphabetic characters

def remove_non_alphabetic(text):
    return ''.join(char for char in text if char.isalpha() or char.isspace())



df2['review_body']=df2['review_body'].apply(remove_non_alphabetic)
df2.head(10)


# In[22]:


import re
def remove_special_symbols(row):
    text=str(row['review_body'])
    text=re.sub("https?:\/\/.*[\r\n]*", " ", text)
    text = re.sub("[#\/,?\!:\$]", " ", text)
    text=text.encode(encoding="ascii", errors="ignore").decode()
    text=re.sub(r'\s+',' ',text)
    ans=re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?","", text)
    return ans

df2['review_body'] = df2.apply (lambda row: remove_special_symbols(row), axis=1)


# In[23]:


#removed special symbols in the above cell


# In[24]:


#removing numbers from reviews
#import warnings
#warnings.filterwarnings('ignore')

#df2["review_body"] = df2["review_body"].str.replace('\d+','')


# In[25]:


#remove extra spaces.
def remove_extra_space(row):
    text=str(row["review_body"])
    ans=" ".join(text.split())
    return ans


# In[26]:


df2['review_body'] = df2.apply (lambda row: remove_extra_space(row), axis=1)


# In[27]:


len_after_cleaning=df2['review_body'].str.len().mean()
print("avg length of review body after cleaning:",len_after_cleaning)


# Here we are printing the average length of the review body after cleaning so that we can compare with the length before cleaning.

# ## 3. Pre-processing

# - Preprocessing involves transforming raw data into a format suitable for analysis or modeling, often including tasks like removing noise, handling missing values, and standardizing text.
# - Stop words are common words (e.g., "the," "and," "is") that are often removed during text preprocessing to reduce noise and focus on the essential content of the text. So here we have removed stop words.
# - Lemmatization is the process of reducing words to their base or root form, helping to standardize different forms of a word. It is used to enhance the effectiveness of natural language processing tasks by reducing variations. We performed lemmatization to get the words standardized form.
# 

# ### removing stop words 

# In[28]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))


# Function to remove stop words

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


# Applying the remove_stopwords function to the 'review_body' column
df2['review_body'] = df2['review_body'].apply(remove_stopwords)



# In[29]:


print(df2)


# ###  Lemmatization

# In[30]:


import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word,pos='v') for word in words]
    words = [lemmatizer.lemmatize(word,pos='n') for word in words]
    words = [lemmatizer.lemmatize(word,pos='a') for word in words]
    words = [lemmatizer.lemmatize(word,pos='r') for word in words]
    words = [lemmatizer.lemmatize(word,pos='s') for word in words]
    return ' '.join(words)


# In[31]:


df2['review_body'] = df2.review_body.apply(lemmatize_text)
print(df2)


# Here again we are just displaying 3 rows and checking the status of our work

# In[32]:


# Displaying 3 sample rows 
sample_reviews = df2.sample(3)
print("3 Reviews:")
print(sample_reviews)


# In[33]:


len_after_preprocessing=df2['review_body'].str.len().mean()
print("avg length of review body after preprocessing:",len_after_preprocessing)


# Here we are checking the average length of the review body after preprocessing and cleaning

# ## 4. TF-IDF Feature Extraction

# - TF-IDF (Term Frequency-Inverse Document Frequency) quantifies the significance of words in a document relative to a document collection. Implementation in Python involves using libraries like scikit-learn, employing TfidfVectorizer to convert raw documents into TF-IDF features.
# - Extracted 8000 features, TF-IDF feature vectors into features_df dataframe.

# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer
features = df2['review_body']
labels = df2['class']

tfidf_vectorizer = TfidfVectorizer(max_features=8000)
tfidf_features = tfidf_vectorizer.fit_transform(features)
features_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
features_df


# In[35]:


#splitting data into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)


# - We have split the data into 80, 20% . 80 for training data and 20 for testing using train_test_split function.
# - A perceptron is a simple neural network algorithm which is used for binary classification, learning a decision boundary to separate two classes based on input features.
# - We used perceptron to train and predict lables for x_test which is y_test _prediction

# ## 5. Perceptron

# In[36]:


from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)


# In[37]:


y_train_predict = perceptron.predict(X_train)
y_test_predict = perceptron.predict(X_test)


# In[38]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_metrics(y_true, y_predict, dataset_type="Training"):
    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)

    print(f"{dataset_type} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

print_metrics(y_train, y_train_predict, "Training for perceptron")

print_metrics(y_test, y_test_predict, "Testing for perceptron")


# ## 6. SVM

# - Linear SVC (Support Vector Classification) is a variant of SVM (Support Vector Machine) for linearly separable data, aiming to find a hyperplane that best divides classes. 
# - We used SVM classifier to the train data, which predicts on the test data, and evaluates the model's performance by printing accuracy, classification report, and confusion matrix.

# In[39]:


from sklearn.svm import LinearSVC


# In[40]:


# Creating an SVM classifier
svm_classifier = LinearSVC()


# In[41]:


svm_classifier.fit(X_train, y_train)


# In[42]:


# Make predictions on the test data
predictions = svm_classifier.predict(X_test)


# In[43]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to print metrics
def print_metrics(y_true, y_predict, dataset_type="Training"):
    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)

    print(f"{dataset_type} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

# Print metrics for training set
print_metrics(y_train, y_train_predict, "Training")

# Print metrics for testing set
print_metrics(y_test, predictions, "Testing")


# ## 7. Logistic Regression

# - Linear Regression involves using a statistical approach to establish a connection between a dependent variable and independent variables by fitting a linear equation to the available data.
# - Below code utilizes Logistic Regression, a classification algorithm, to train a model on the training data, make predictions on the test data, and then assess the model's performance by printing accuracy, classification report, and confusion matrix.
# - obtained accuracy, precision, recall and F1 score metrics 

# In[44]:


from sklearn.linear_model import LogisticRegression
# Create a Logistic Regression classifier
logistic_classifier = LogisticRegression(random_state=42)


# Train the Logistic Regression classifier on the training data
logistic_classifier.fit(X_train, y_train)


# Make predictions on the test data
predictions = logistic_classifier.predict(X_test)




# In[45]:


def print_metrics(y_true, y_predict, dataset_type="Testing"):
    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)

    print(f"{dataset_type} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    
# Make predictions on the training data
predictions_train = logistic_classifier.predict(X_train)

# Evaluate accuracy for training set
accuracy_train = accuracy_score(y_train, predictions_train)
print(f'Logistic Regression Training Accuracy: {accuracy_train:.2f}')

# Print all metrics for training set
print_metrics(y_train, predictions_train, "Training")

# Evaluate accuracy for testing set
accuracy_logistic = accuracy_score(y_test, predictions)
print(f'Logistic Regression Testing Accuracy: {accuracy_logistic:.2f}')

# Print all metrics for testing set
print_metrics(y_test, predictions, "Testing")




# ## 8. Naive Bayes

# - Naive Bayes is a probabilistic classification algorithm that applies Bayes' theorem, assuming feature independence. It finds common use in tasks like text classification and spam filtering.
# - Initialization involves creating a Multinomial Naive Bayes model using scikit-learn's MultinomialNB().
# - Training the model on the provided training data (X_train, y_train) is done through the fit method.
# - Predictions on both training and testing sets (X_train, X_test) are made, followed by the printing of metrics such as accuracy, precision, recall, and F1-score using a custom function (print_metrics).

# In[46]:


from sklearn.naive_bayes import MultinomialNB
naivebayes_model = MultinomialNB()
naivebayes_model.fit(X_train, y_train)


# In[47]:


y_train_mnb_predict = naivebayes_model.predict(X_train)
y_test_mnb_predict = naivebayes_model.predict(X_test)


# In[48]:


print("Naive Bayes")

print_metrics(y_train, y_train_mnb_predict, "Training")

# Printing metrics for testing set

print_metrics(y_test, y_test_mnb_predict, "Testing")


# In[ ]:




