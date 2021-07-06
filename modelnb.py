# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, plot_precision_recall_curve, make_scorer, recall_score, brier_score_loss, precision_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# libraries for cleaning
import re
import nltk
nltk.download("stopwords") # helps us get rid of stop words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#csv file load
df = pd.read_csv("text_preprocessed.csv")

df.info()
df.head()

df.loc[df.label == 1,:].head()

def cleaning_words(phrase):
  tweet = re.sub(r"http\S+", "", phrase) # remove all URLs
  tweet = re.sub('[^a-zA-z]',' ',tweet) # remove punctuation
  tweet = re.sub(r'@\S+|https?://\S+','', tweet) # remove @ sign
  tweet = tweet.lower() # make all letters lower case
  #tweet = tweet.split() # make a list of the words

  # now will stem words
  # ps = PorterStemmer()
  # all_stopwords = stopwords.words("english")
  # all_stopwords.remove("not") # make sure we don't remove the word "not" since that changes the meaning of the sentence
  # tweet = [ps.stem(t) for t in tweet if not t in set(all_stopwords)]
  # tweet = " ".join(tweet) # join the words back together
  cleaned_words = []
  cleaned_words.append(tweet)
  return cleaned_words

# split into training vs testing data

X = df.text
y = df.label
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
#X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.25, random_state=0)

# create a sparse matrix
cv = CountVectorizer(stop_words=stopwords.words("english"))
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
#X_val = cv.transform(X_val)
feature_names = cv.get_feature_names()

print(feature_names[5000:5010])

# a ton of unique words! (166095 to be exact)
X_train.shape

from sklearn.naive_bayes import MultinomialNB

# find baseline accuracy
fake_count = df[df.label == 1].count()[0]
real_count = df[df.label == 0].count()[0]

perc_f = fake_count/df.shape[0]
perc_r = real_count/df.shape[0]

print(f"{perc_f} of the data is fake")
print(f"{perc_r} of the data is real")

from sklearn.model_selection import GridSearchCV

# fit the model
model_mb = MultinomialNB()
params = {"alpha":[0.001, 0.01, 0.05, 0.1, 0.5]}

model_mb_grid = GridSearchCV(model_mb, params,n_jobs=1, verbose = 1, scoring = "f1")
model_mb_grid.fit(X_train,y_train)
best_params = model_mb_grid.best_params_
print(best_params)

# alpha = model_mb_grid.best_params_.get('alpha')
model_mb = MultinomialNB()
model_mb.fit(X_train, y_train)

# find performance on training vs test data
y_test_pred = model_mb.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))

# since NB also gives probabilities, we can calculate the Brier Score
y_train_pred_prob = model_mb.predict_proba(X_train)[:,1]
y_test_pred_prob = model_mb.predict_proba(X_test)[:,1]

brier_train = brier_score_loss(y_train, y_train_pred_prob)
brier_test = brier_score_loss(y_test, y_test_pred_prob)
print(brier_train)
print(brier_test)


#import library
import pickle
from pickle import dump

#make pickle file of our model
pickle.dump(model_mb, open("model.pkl", "wb"))
pickle.dump(cv, open('count.pkl', 'wb'))

