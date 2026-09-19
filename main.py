import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from keras.utils import pad_sequences
import pickle

### load model
model = load_model('model.h5')
with open('tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)

st.title("Twitter Tweets Sentiment Analysis")

tweet = st.text_area('Enter the Tweet')

if st.button('Predict Sentiment') and tweet.strip():
    sequences = tokenizer.texts_to_sequences(tweet)
    sequences=pad_sequences(sequences,maxlen=166,padding='pre')
    prediction = model.predict(sequences)
    pred = np.argmax(prediction,axis=1)[0]
    sentiment_map ={0:'Negative',1:'Neurtal',2:'Positive'}
    st.write('Sentiment:',sentiment_map[pred])