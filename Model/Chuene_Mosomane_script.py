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
display(df.head())

# Display the summary statistics
display(df.describe())

# Print the info
print(df.info())

# convert all words to lower case
df.dirty_name = df.dirty_name.apply(lambda x: x.lower())

# Check for differnt labels
pd.unique(df.dirty_label)

# How many words do we have per each dirty_labels?

person = df[df.dirty_label== 'Person']
company = df[df.dirty_label== 'Company']
university = df[df.dirty_label== 'University']

#person.shape, university.shape, company.shape

# Check how dirty_labels are distributed

# Print the counts of each labels
print(df['dirty_label'].value_counts())

# Print the proportions of each labels
print(df['dirty_label'].value_counts(normalize=True))

# Visualize the labels
sns.countplot(df['dirty_label'])
plt.title("label Counts")
plt.show()

# How many words do we have per each dirty_labels?

person = df[df.dirty_label== 'Person']
company = df[df.dirty_label== 'Company']
university = df[df.dirty_label== 'University']

# Store the number of words in each names
df['word_count'] = df['dirty_name'].str.split().str.len()


# Visualize the distribution of word counts in each label
sns.distplot(df[df['dirty_label']=='Person']['word_count'], label='Person')
sns.distplot(df[df['dirty_label']=='Company']['word_count'], label='Company'),
sns.distplot(df[df['dirty_label']=='University']['word_count'], label='University'),
plt.legend()
plt.show()

##################################################################################################################################
# Data prepreocessing
#################################################################################################################################

# create a function to count words
def word_count_fun(dictionary):
    list_vec = list(dictionary.keys())
    full_list = []
    for x in list_vec:
        x1 = x.split()
        for x2 in x1:
            full_list.append(x2)
    return  full_list

# create dataframes for person, company, university
vectorizer_per = dict(Counter(person.dirty_name.to_list()))
vectorizer_com = dict(Counter(company.dirty_name.to_list()))
vectorizer_uni = dict(Counter(university.dirty_name.to_list()))

count_per =pd.DataFrame(list(dict(Counter(word_count_fun(vectorizer_per))).items()),columns = ['names','counting'])\
                                    .sort_values("counting", ascending=False)

count_com =pd.DataFrame(list(dict(Counter(word_count_fun(vectorizer_com))).items()),columns = ['names','counting'])\
                                    .sort_values("counting", ascending=False)

count_uni =pd.DataFrame(list(dict(Counter(word_count_fun(vectorizer_uni))).items()),columns = ['names','counting'])\
                                    .sort_values("counting", ascending=False)

# Here I created this manually, to verify the labels.
# So later I create a mapping fuction using these lists as my inputs

university_list = ['université', 'university', 'universitas', 'universitaria', 'universidade', 'universidad',\
                   'tecnológica', 'technology','science', 'school', "medicine",'instituto', 'institute',
                  "health", 'educativas', 'education', 'academy']
company_list = ['ltd', 'pty', 'pl', 'co.', 'cc', 'ltd.', 'limited', 'gmbh','trust', 'co', 'fund', '(pty)', \
                'group', 'family', 'inc.','company', 'consortium','inc', 'c.c.', 'limited.', 'unltd', 'cc.', \
                'llc.', 'corp.', 'funds', 'pllc.', 'services', 'partnership', 'corp', '(ltd).', 'trading',\
               "proprietary", 'lda.', 'lda', 'investments', 'trudoo', '(ltd)', 'capital',  'associates', \
                "sicav", 'plc.', 'plc', 'incorporated', 'l.l.p',]

person_list = ['miss', 'dr.', 'prof.', 'dr', 'mr.', 'mr', 'mrs', 'mrs.', 'van','hon.', 'rev.', 'ms', 'sr.', \
               'ms.']

# create a map function to verify the labels of dirty_label
def mapping_func(dirtyname):
    return_value = ''
    for i in dirtyname.split():
        #print(i)
        if i in university_list:
            return_value =  "University"
            break
        elif i in company_list:
            return_value = "Company"
            break
        elif not return_value:
            return_value = "Person"
        
    return return_value

# Create a new column, semi clean label
df["semi_clean_label"] = df.dirty_name.apply(mapping_func)
#################################################################################################################################################
# Split data
#################################################################################################################################################

# Select the features and the label for dirty label
X = df['dirty_name']
y = df['dirty_label']

# Select the features and the label for semi clean label
X_ = df['dirty_name']
y_ = df['semi_clean_label']

# Split the data for the dirty label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

# Split the data for the semi clean label
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.2, random_state=34)

# Create the tf-idf vectorizer
vectorizer = TfidfVectorizer()

# First fit the vectorizer with our training set
tfidf_train = vectorizer.fit_transform(X_train)

# Now we can fit our tes(dirty label)t data with the same vectorizer
tfidf_test = vectorizer.transform(X_test)

# We also fit our  test(semi clean label) with the vectorizer
tfidf_test_ = vectorizer.transform(y_test_)

#####################################################################################################################################################
# Build Model
####################################################################################################################################################

# Initialize the Multinomial Naive Bayes classifier
model_naive_bayes = MultinomialNB()

# Fit the model using the dirty label y_train 
model_naive_bayes.fit(tfidf_train, y_train)

# Print the accuracy score
print("Accuracy:", model_naive_bayes.score(tfidf_test, y_test))

##############################################################################################################################################
# Evaluate Model
#############################################################################################################################################
# Predict using the semi clean label
y_pred = model_naive_bayes.predict(tfidf_test_)

# Print the Confusion Matrix
cm = confusion_matrix(y_test_, y_pred)
print("Confusion Matrix\n")
print(cm)

# Print the Classification Report
cr = classification_report(y_test_, y_pred)
print("\n\nClassification Report\n")
print(cr)

# Print the Receiver operating characteristic Auc score
#auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr')
accuracy = accuracy_score(y_test_, y_pred)
print("Accuracy Score:",accuracy)
####################################################################################################################################
# Save Model
####################################################################################################################################

import pickle
# open a file, where you ant to store the data
file = open('Naive_Bayes_Classifier.pkl', 'wb')

# dump information to that file
pickle.dump(model_naive_bayes, file)

# Load the model
model = pickle.load(open('Naive_Bayes_Classifier.pkl', 'rb'))