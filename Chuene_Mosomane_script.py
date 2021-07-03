# Fundamentals
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Natural Language Processing
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

# Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

# Import Tf-idf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the Label Encoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# Import the train test split
from sklearn.model_selection import train_test_split

# To evaluate our model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn import metrics
#################################################################################################################################
# Load data
df = pd.read_csv('names_data_candidate.csv')
##################################################################################################################################

###################################################################################################################################
# Exploratory Data Analysis
###################################################################################################################################

# Display firt five rows
df.head()

# Display the summary statistics
df.describe()

# Print the info
df.info()

# Print the counts of each labels
 count_labels = df['dirty_label'].value_counts()

#print(count_labels)

# Print the proportions of each labels
proportion_labels = df['dirty_label'].value_counts(normalize=True))

#print(proportion_labels)

# Visualize the Categories
sns.countplot(df['dirty_label'])
plt.title("label Counts")
#plt.show()

# Initialize the Label Encoder.
le = LabelEncoder()

# Encode the labels
df['dirty_label_enc'] = le.fit_transform(df['dirty_label'])

# Display the first five rows again to see the result
df.head()

# Print the datatypes
df.dtypes

# Store the number of words in each names
df['word_count'] = df['dirty_name'].str.split().str.len()

# Print the average number of words in each label
print(df.groupby('dirty_label')['word_count'].mean())

# Visualize the distribution of word counts in each label
sns.distplot(df[df['dirty_label']=='Person']['word_count'], label='Person')
sns.distplot(df[df['dirty_label']=='Company']['word_count'], label='Company'),
sns.distplot(df[df['dirty_label']=='University']['word_count'], label='University'),
plt.legend()
plt.show()

##################################################################################################################################
# Data prepreocessing
#################################################################################################################################

# Make the letters lower case and tokenize the words
tokenized_names = df['dirty_name'].str.lower().apply(word_tokenize)

# Print the tokens to see how it looks like
print(tokenized_names)

# Define a function to returns only alphanumeric tokens
def alpha(tokens):
    """This function removes all non-alphanumeric characters"""
    alpha = []
    for token in tokens:
        if str.isalpha(token) or token in ['n\'t','won\'t']:
            if token=='n\'t':
                alpha.append('not')
                continue
            elif token == 'won\'t':
                alpha.append('wont')
                continue
            alpha.append(token)
    return alpha

# Apply our function to tokens
tokenized_names = tokenized_names.apply(alpha)

print(tokenized_names)

# Define a function to remove stop words
def remove_stop_words(tokens):
    """This function removes all stop words in terms of nltk stopwords"""
    no_stop = []
    for token in tokens:
        if token not in stopwords.words('english'):
            no_stop.append(token)
    return no_stop

# Apply our function to tokens
tokenized_names = tokenized_names.apply(remove_stop_words)

print(tokenized_names)

# Define a function to lemmatization
def lemmatize(tokens):
    """This function lemmatize the names"""
    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    # Create the lemmatized list
    lemmatized = []
    for token in tokens:
            # Lemmatize and append
            lemmatized.append(lemmatizer.lemmatize(token))
    return " ".join(lemmatized)

# Apply our function to tokens
tokenized_names = tokenized_names.apply(lemmatize)

print(tokenized_names)

# Replace the columns with tokenized messages
df['dirty_name'] = tokenized_names

# Display the first five rows
display(df.head())

#################################################################################################################################################
# Split data
#################################################################################################################################################

# Select the features and the target
X = df['dirty_name']
y = df['dirty_label_enc']

# Create the tf-idf vectorizer
vectorizer = TfidfVectorizer(strip_accents='ascii')

# First fit the vectorizer with our training set
tfidf_train = vectorizer.fit_transform(X_train)

# Now we can fit our test data with the same vectorizer
tfidf_test = vectorizer.transform(X_test)

#####################################################################################################################################################
# Build Model
####################################################################################################################################################

# Initialize the Multinomial Naive Bayes classifier
model_naive_bayes = MultinomialNB()

# Fit the model
model_naive_bayes.fit(tfidf_train, y_train)

# Print the accuracy score
print("Accuracy:",model_naive_bayes .score(tfidf_test, y_test))

# Predict the labels
y_pred = model_naive_bayes.predict(tfidf_test)

# Print the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n")
print(cm)

# Print the Classification Report
cr = classification_report(y_test, y_pred)
print("\n\nClassification Report\n")
print(cr)

# Print the Receiver operating characteristic Auc score
#auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr')
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:",accuracy)

# Function for calculating roc_auc_score for multi-class
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred,)

multiclass_roc_auc_score(y_test, y_pred)

####################################################################################################################################
# Save Model
####################################################################################################################################

import pickle
# open a file, where you ant to store the data
file = open('Naive_Bayes_Classifier.pkl', 'wb')

# dump information to that file
pickle.dump(model_naive_bayes, file)