# CAN AN INTELLIGENT CHATBOT PROVIDE EFFECTIVE MENTAL HEALTH SUPPORT USING NATURAL LANGUAGE PROCESSING (NLP)?

 Welcome to my "Virtual Mental Chatbot" project! This chatbot is designed to provide general mental health support using natural language processing (NLP) techniques. It can respond to user queries in a way that mimics human interactions, offering helpful information and guidance related to mental health topics. 

**Note**: This chatbot is an experimental project created for educational purposes. It provides general information and support based on predefined responses and NLP techniques.

**Disclaimer**: This chatbot is not a substitute for professional mental health care. For serious concerns, please seek assistance from a licensed mental health professional.

## Table of Contents
  Project Overview
  Features
  Research Objectives
  Installation
  Usage
  Technologies and Libraries Used
  Project Structure
  Limitations
  Acknowledgements

## Project Overview
  This project aims to create an intelligent chatbot capable of providing mental health support through Natural Language Processing (NLP). It involves collecting a dataset of question-answer pairs related to mental health inquiries, training a machine learning model, and deploying it on a user-friendly web interface built with Streamlit.

## Features
- Responds to user queries related to mental health in real-time.
- Trained on a dataset with 98 entries containing various question-answer pairs.
- Uses NLP techniques to understand and respond to user input.
- Accessible via a web interface powered by Streamlit.

## Research Objectives
1. **Dataset Collection**: Gather a dataset containing typical question-answer pairs related to mental health.
2. **Data Preparation**: Preprocess the data to make it suitable for training an NLP model.
3. **Model Development**: Develop and train a deep learning model to generate human-like responses to mental health questions.
4. **Performance Evaluation**: Assess the model's performance in terms of accuracy, responsiveness, and reliability.
5. **Deployment**: Implement the trained model in a Streamlit-based web application for real-time use.

## Installation
 # Navigate to the Project Directory:
  cd "C:\Users\HP\OneDrive - University of Hertfordshire\Virtual Mental Chatbot"

 # Set Up a Virtual Environment:
  ## Install Virtual Environment
  pip install virtualenv

 # Create a Virtual Environment
  virtualenv venv
  
 # Activate Virtual Environment
  venv\Scripts\activate

 # Install Requirements
 - pip install tensorflow 
  
 - pip install streamlit 
  
 - pip install numpy 
  
 - pip install keras 
  
 - pip install nltk 
  
## Usage  
  ## Train the Model
 - Run the training.py script to process data and train the model.
  **python training.py**

 # Details:
 - Data Preparation: Tokenizes and lemmatizes text data.
 - Model Training: Builds and trains a neural network with Keras.
 - Output: Saves the trained model to model.chatbot.keras.

## Run the Application
 - Launch the Streamlit application to interact with the chatbot.
  **streamlit run app.py**
 - This will open a new tab in my default web browser where I can interact with the chatbot.  

 # Details:
 - Interaction: Uses app.py to provide a web-based interface for chatting with the bot.
 - Functionality: Processes user input, predicts intents, and generates responses.

## Access my Virtual Chatbot
  The chatbot interface will be available at  http://localhost:8501/ in my web browser


## Technologies and Libraries Used 
 The following libraries are used in this project to build the "Virtual Mental Chatbot":

 1. **NLTK (Natural Language Toolkit)**: Used for natural language processing tasks such as tokenization and lemmatization to preprocess user inputs and patterns, ensuring accurate intent matching.

 2. **JSON**: Used to load and parse the intents.json file, which contains the defined intents, example patterns, and corresponding responses the chatbot uses.

 3. **Pickle**: Used to save and load serialized Python objects (like words and classes) to speed up the chatbot's initialization process.

 4. **NumPy**: Provides efficient handling and manipulation of numerical data for creating the "bag of words" model and preparing data for the neural network.

 5. **Keras**: Used to define and train the neural network model that predicts the user's intent and selects the appropriate response.

 6. **Random**: Adds variability to the chatbot's responses by selecting random responses from predefined options, making interactions feel more natural.

 7. **Streamlit**: Provides a web-based user interface, allowing users to interact with the chatbot in real-time.

 8. **TensorFlow**: Serves as the backend engine for Keras, handling neural network computations and optimizations to power the chatbot's machine learning capabilities.

## Project Structure
 - venv: Virtual environment directory.
 - app.py: The main application script that launches the chatbot using Streamlit.
 - training.py: Script to preprocess data and train the NLP model.
 - intents.json: Contains predefined intents, patterns, and responses used by the chatbot.
 - data.json: Additional data for model training/testing.
 - mental health FAQ.csv: Dataset containing frequently asked questions related to mental health.
 - mentaltrain.ipynb: Jupyter Notebook for experimental development.
 - model.h5: Directory containing trained model files
 - texts.pkl: Pickled file containing tokenized words.
 - labels.pkl: Pickled file containing output labels.

## Limitations
 - Scope: The chatbot provides general responses and may not address complex or specific mental health issues effectively.
 - Dataset Size: The chatbot is trained on a relatively small dataset of 98 entries. This may limit its ability to handle a wide range of mental health topics and questions accurately.
 - Accuracy: The model's performance depends on the quality and coverage of the training data.
 - Not a Replacement for Professional Help: The chatbot is designed for educational and informational purposes only and is not a substitute for professional mental health care. Users seeking serious or urgent mental health support should consult a licensed professional.

## Acknowledgements
 I would like to express my sincere gratitude to the following individuals and organizations for their support and contributions to my **Virtual Mental Chatbot** project:
 - **Michael Walters**, my project supervisor, for their invaluable guidance, insightful feedback, and encouragement throughout the development of my project.
 - **University of Hertfordshire**, for providing the necessary resources and access to facilities that were crucial for the successful completion of my project.
 - **Libraries and Tools**, including NLTK, Keras, TensorFlow, and Streamlit, for their powerful tools and libraries that were instrumental in developing and deploying the chatbot

 Your support and contributions have been greatly appreciated and were vital to the success of my project.