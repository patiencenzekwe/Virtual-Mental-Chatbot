# mental-health-chatbot
## CAN AN INTELLIGENT CHATBOT PROVIDE EFFECTIVE MENTAL HEALTH SUPPORT USING NATURAL LANGUAGE PROCESSING (NLP)

## Table of Contents
  Project Overview
  Research Aim
  Features
  Technologies Used
  Installation
  Usage
  Project Structure
  Limitations
  Contribution
  License
  Acknowledgements

## Project Overview
  This project involves the development of a virtual mental health chatbot that uses Natural Language Processing (NLP) to respond to mental health-related queries. The chatbot aims to provide general information and supportive responses, simulating human-like interactions to assist users with mental health inquiries.

## Disclaimer: 
  This chatbot is intended for educational purposes only and is not a substitute for professional mental health services.

## Research Objectives
  Dataset Collection: Gather and compile question-answer pairs related to mental health from various sources.
  Data Preparation: Clean and format the dataset to train NLP models effectively.
  Model Development: Train an NLP model to generate human-like responses based on the dataset.
  Evaluation: Assess the model's accuracy, responsiveness, and reliability in handling mental health queries.
  Deployment: Implement the trained model in a Streamlit-based web application for real-time interaction.

## Features
  Interactive Chatbot: Provides responses to various mental health inquiries.
  Human-like Conversations: Uses machine learning to generate natural responses.
  Web Interface: Streamlit-based application for easy user interaction.

## Technologies Used
  Programming Language: Python
  Libraries and Frameworks:
  NLP: NLTK
  Machine Learning: Keras, TensorFlow
  Data Handling: NumPy, Pandas, JSON, Pickle
  Web Application: Streamlit

## Installation
 ## Navigate to the Project Directory:
  cd "C:\Users\HP\OneDrive - University of Hertfordshire\Virtual Mental Chatbot"

 ## Set Up a Virtual Environment:
  ## Install Virtual Environment
  pip install virtualenv

  ## Create a Virtual Environment
  virtualenv venv
  
  ## Activate Virtual Environment
  venv\Scripts\activate

  ## Install Requirements
  pip install tensorflow 
  
  pip install streamlit 
  
  pip install numpy 
  
  pip install keras 
  
  pip install nltk 
  
## Usage  
  ## Train the Model
  Run the training.py script to process data and train the model.
  python training.py

  Details:
  Data Preparation: Tokenizes and lemmatizes text data.
  Model Training: Builds and trains a neural network with Keras.
  Output: Saves the trained model to model.chatbot.keras.

## Run the Application
  Launch the Streamlit application to interact with the chatbot.
  streamlit run app.py
  This will open a new tab in my default web browser where I can interact with the chatbot.  

  Details:
  Interaction: Uses app.py to provide a web-based interface for chatting with the bot.
  Functionality: Processes user input, predicts intents, and generates responses.

## Access my Virtual Chatbot
  The chatbot interface will be available at  http://localhost:8501/ in my web browser

## Project Structure
  venv - Virtual environment directory
  data.json - Additional dataset for training/testing
  intents.json - Contains intent definitions, patterns, and responses
  labels.pkl - Pickled file for storing intent labels
  mental health FAQ.csv - Dataset containing frequently asked questions related to mental health
  mentaltrain.ipynb - Jupyter Notebook for experimental development
  model.h5 - Directory containing trained model files
  texts.pkl - Pickled file for storing tokenized text data
  training.py - Script for training the NLP model
  app.py - Main script for the Streamlit web application

## Limitations
  Scope: The chatbot provides general responses and may not address complex or specific mental health issues effectively.
  Dataset Size: Limited to the dataset used for training, which may impact the chatbot's ability to handle diverse queries.
  Accuracy: The model's performance depends on the quality and coverage of the training data.

## Acknowledgements
  Thanks to the developers of NLTK, Keras, TensorFlow, and Streamlit.
  Appreciation for mental health resources and professionals.
  Thanks to University of Hertfordshire

# Congatulations!!
