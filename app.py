import streamlit as st
import random
import json
from keras.models import load_model
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK popular packages (only needed if not already downloaded)
nltk.download('popular')

# Initialize the lemmatizer and load model, intents, words, and classes
lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')
intents = json.loads(open('intents.json', "r+", encoding="utf-8").read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Define functions for processing input and generating responses
def clean_up_sentence(sentence):
    # Tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word - create base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    # Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    else:
        result = "Sorry, I didn't understand that. Can you please rephrase?"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Streamlit application
st.title("Virtual Mental Chatbot")
st.write("Hello and welcome! I am a virtual mental health chatbot here to assist you. Feel free to message me, and I will do my best to help. However, please note that I can only provide general information and respond to specific questions. I am not a licensed professional, so for serious concerns, please seek guidance from a qualified mental health professional.")

# Text input for user
user_input = st.text_input("You:", "")

if user_input:
    response = chatbot_response(user_input)
    st.text_area("Bot:", value=response, height=100, max_chars=None, key=None)
