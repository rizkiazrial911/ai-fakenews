import string

import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Read data
truenews = pd.read_csv('news_true.csv')
fakenews = pd.read_csv('news_fake.csv')
truenews['True/Fake'] = 'True'
fakenews['True/Fake'] = 'Fake'

# Combine the 2 DataFrames into a single data frame
news = pd.concat([truenews, fakenews])
news["Article"] = news["title"] + news["text"]
news.sample(frac=1)  # Shuffle 100%

# Data Cleaning
nltk.download('stopwords')


def process_text(s):
    # Check string to see if they are punctuation
    nopunc = [char for char in s if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # Convert string to lowercase and remove stopwords
    clean_string = [word for word in nopunc.split() if word.lower() not in stopwords.words('indonesian')]
    return clean_string

# Tokenize the text :Convert the normal text strings in to a list of tokens (words that we actually want)
#rerun, takes LOOOONG
news['Clean Text'] = news['Article'].apply(process_text)

bow_transformer = CountVectorizer(analyzer=process_text).fit(news['Clean Text'])

#Bag-of-Words (bow) transform the entire DataFrame of text
news_bow = bow_transformer.transform(news['Clean Text'])

sparsity = (100.0 * news_bow.nnz / (news_bow.shape[0] * news_bow.shape[1]))

tfidf_transformer = TfidfTransformer().fit(news_bow)
news_tfidf = tfidf_transformer.transform(news_bow)

#Train Naive Bayes Model
fakenews_detect_model = MultinomialNB().fit(news_tfidf, news['True/Fake'])

#Model Evaluation
predictions = fakenews_detect_model.predict(news_tfidf)

# Tokenize the text
news_train, news_test, text_train, text_test = train_test_split(news['Article'], news['True/Fake'], test_size=0.3)

# Train Naive Bayes Model
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(news_train,text_train)


# Streamlit app
st.title("Fake News Detection App")

# Input text box for user input
user_input = st.text_input("Enter the news text:")

if user_input:
    # Predict using the trained model
    prediction = pipeline.predict([user_input])

    # Display the result
    st.subheader("Prediction:")
    st.write(f"The input news is classified as: {prediction[0]}")
