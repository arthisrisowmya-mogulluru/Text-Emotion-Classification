# Importing Libraries
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings('ignore')

!pip install fasttext
import fasttext
import pickle

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout

# Data Source
# Collected a publicly available dataset from [source], which contains user comments and labels for emotions.
# Read data
train = pd.read_csv("/content/drive/MyDrive/train.txt", delimiter=';', header=None, names=['sentence', 'label'])
test = pd.read_csv("/content/drive/MyDrive/test.txt", delimiter=';', header=None, names=['sentence', 'label'])

df = pd.concat([train, test])
print(df.shape)
df.head()

print(df['label'].value_counts())
print('\n')
print(df['label'].value_counts(normalize=True))

print(train.shape)
print(train['label'].value_counts(normalize=True))

print(test.shape)
print(test['label'].value_counts(normalize=True))

# Preprocess Text
# Data Preprocessing
# Data cleaning involved removing noise, handling missing values, and ensuring consistent formatting.
# Text data was tokenized, and stopwords were removed.

def clean_data(data):
    stop_words = set(stopwords.words('english'))

    # Create a new DataFrame to avoid modifying the original data
    new_data = pd.DataFrame({'sentence': data['sentence'], 'label': data['label']})

    # Clean the 'sentence' column
    new_data['sentence'] = new_data['sentence'].apply(
        lambda sentence: ' '.join(w.lower() for w in sentence.split() if (w.lower() not in stop_words) and w.isalpha()))

    # Remove rows with empty 'sentence' values
    new_data = new_data[new_data['sentence'] != '']

    # Reset the index
    new_data = new_data.reset_index(drop=True)

    return new_data

def extract_features(train_set, test_set, ngram):
    tfidf = TfidfVectorizer(use_idf=True, max_df=0.95, ngram_range=ngram)
    tfidf.fit_transform(train_set['sentence'].values)

    train_tfidf = tfidf.transform(train_set['sentence'].values)
    test_tfidf = tfidf.transform(test_set['sentence'].values)

    return train_tfidf, test_tfidf, tfidf

df_clean = clean_data(df)
df_clean.to_csv('/content/drive/MyDrive/data_clean.csv')

training_df, testing_df = train_test_split(df_clean[['sentence', 'label']].dropna(),
                                           test_size=0.2, random_state=2020)

X_train, X_test, tfidf_vectorizer = extract_features(training_df, testing_df, (1, 2))
y_train = training_df['label'].values
y_test = testing_df['label'].values

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

np.mean([len(i) for i in df_clean.sentence])

# Logistic Regression
# Logistic Regression as a simple baseline model for emotion classification.
# Feature vectors were created using TF-IDF representation.

def predict_evaluate(model, test_tfidf, test_y):
    prediction = model.predict(test_tfidf)
    print(classification_report(test_y, prediction))
    return prediction

lr1 = LogisticRegression(random_state=2020, C=15, penalty='l2', max_iter=1000, verbose=1)
lr1_classifier = OneVsRestClassifier(lr1)
model_lr1 = lr1_classifier.fit(X_train, y_train)

lr1_pred = predict_evaluate(model_lr1, X_test, y_test)

lr2 = LogisticRegression(random_state=2020, C=5, penalty='l2', max_iter=1000, verbose=1)
lr2_classifier = OneVsRestClassifier(lr2)
model_lr2 = lr2_classifier.fit(X_train, y_train)

lr2_pred = predict_evaluate(model_lr2, X_test, y_test)

lr3 = LogisticRegression(random_state=2020, C=30, penalty='l2', max_iter=1000, verbose=1)
lr3_classifier = OneVsRestClassifier(lr3)
model_lr3 = lr3_classifier.fit(X_train, y_train)

lr3_pred = predict_evaluate(model_lr3, X_test, y_test)

filepath = "logistic_best.pkl"
with open(filepath, 'wb') as file:
    pickle.dump(model_lr3, file)

# SVM
# Support Vector Machine (SVM) for emotion classification.
# Feature vectors were generated using TF-IDF encoding, similar to the Logistic Regression approach.

svm1 = LinearSVC(random_state=2020, C=1, loss='squared_hinge', max_iter=1000)
model_svm1 = svm1.fit(X_train, y_train)

svm1_pred = predict_evaluate(model_svm1, X_test, y_test)

svm2 = LinearSVC(random_state=2020, C=50, loss='squared_hinge', max_iter=1000)
model_svm2 = svm2.fit(X_train, y_train)

svm2_pred = predict_evaluate(model_svm2, X_test, y_test)

svm3 = LinearSVC(random_state=2020, C=50, loss='hinge', max_iter=1000)
model_svm3 = svm3.fit(X_train, y_train)

svm3_pred = predict_evaluate(model_svm3, X_test, y_test)

svm4 = LinearSVC(random_state=2020, C=1, loss='hinge', max_iter=1000)
model_svm4 = svm4.fit(X_train, y_train)

svm4_pred = predict_evaluate(model_svm4, X_test, y_test)

filepath = "svm_best.pkl"
with open(filepath, 'wb') as file:
    pickle.dump(model_svm2, file)

# FastText
# Using the FastText library for sentiment analysis.

# Format the data
train_fasttext = training_df.apply(lambda t: '__label__' + str(t['label']) +
                                   ' ' + str(t['sentence']), axis=1)
test_fasttext = testing_df.apply(lambda t: '__label__' + str(t['label']) +
                                 ' ' + str(t['sentence']), axis=1)
train_fasttext.to_csv('train_fasttext.txt', index=False, header=False)
test_fasttext.to_csv('test_fasttext.txt', index=False, header=False)

model_ft1 = fasttext.train_supervised('train_fasttext.txt', loss='softmax',
                                      lr=0.1, ws=5, wordNgrams=2, epoch=100)
ft1_pred = model_ft1.test('test_fasttext.txt')

print("precision: ", ft1_pred[1])
print("recall: ", ft1_pred[2])
print("F-1 score: ", 2 * ft1_pred[1] * ft1_pred[2] / (ft1_pred[1] + ft1_pred[2]))

model_ft12 = fasttext.train_supervised('train_fasttext.txt', loss='softmax',
                                      lr=0.2, ws=10, wordNgrams=2, epoch=100)
ft2_pred = model_ft12.test('test_fasttext.txt')

print("precision: ", ft2_pred[1])
print("recall: ", ft2_pred[2])
print("F-1 score: ", 2 * ft2_pred[1] * ft2_pred[2] / (ft2_pred[1] + ft2_pred[2]))

model_ft13 = fasttext.train_supervised('train_fasttext.txt', loss='softmax',
                                      lr=0.2, ws=10, wordNgrams=2, epoch=300)
ft3_pred = model_ft13.test('test_fasttext.txt')

print("precision: ", ft3_pred[1])
print("recall: ", ft3_pred[2])
print("F-1 score: ", 2 * ft3_pred[1] * ft3_pred[2] / (ft3_pred[1] + ft3_pred[2]))

model_ft14 = fasttext.train_supervised('train_fasttext.txt', loss='softmax',
                                      lr=0.2, ws=15, wordNgrams=2, epoch=100)
ft4_pred = model_ft14.test('test_fasttext.txt')

print("precision: ", ft4_pred[1])
print("recall: ", ft4_pred[2])
print("F-1 score: ", 2 * ft4_pred[1] * ft4_pred[2] / (ft4_pred[1] + ft4_pred[2]))
