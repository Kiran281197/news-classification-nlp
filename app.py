import streamlit as st
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

#load model, tokenizer and label_mapper
model = load_model("bi_lstm.h5")

with open("tokenizer.pkl","rb") as file:
    tokenizer = pickle.load(file)

with open("label_mapper.pkl","rb") as file:
    label_mapper = pickle.load(file)

max_sequence_len = 70


st.title("News Classifier")

# remove punctuations
def clean_txt(text):
    text_lower = text.strip().lower()
    clean_txt = re.sub(r"[^\w\s]","",text)
    return clean_txt

def predict_news(txt):
    text = clean_txt(txt)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_len, padding="post")
    prediction = model.predict(padded_sequence)
    prediciton_index = np.argmax(prediction)
    return prediciton_index


news = st.text_input("Enter the News Headlines")


if st.button("Submit"):
    if news!="":
        prediction_index = predict_news(news)
        st.success(f"{label_mapper[prediction_index]}")
    else:
        st.error("Please enter the news headlines")
    

