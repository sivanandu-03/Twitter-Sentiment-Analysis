import os
import pickle
import streamlit as st

if os.path.exists('trained_model.sav'):
    print("Model file found!")
else:
    print("Model file not found!")


sentiment_model = pickle.load(open('trained_model.sav','rb'))


st.title('Twitter Sentiment Analysis')
tweet=st.text_input('Enter The Tweet')
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
port_stem = PorterStemmer()
def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

final_tweet = stemming(tweet)
vectorizer = TfidfVectorizer()
vectorizer.fit(final_tweet)
final_tweet = vectorizer.transform(final_tweet)

result = ''
if st.button('Checking Sentiment'):
    sentiment_prediction = sentiment_model.predict(final_tweet)
    if(sentiment_prediction[0]==0):
        result = 'Negative Tweet'
    else:
        result = 'Positive Tweet'
st.success(result) 
