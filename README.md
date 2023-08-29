
# Breast Cancer Diagnosis using Machine Learning
https://github.com/hemanitekwani/ml-nlp_projects/blob/main/Breast_cancer.py

This repository contains code for a machine learning project focused on breast cancer diagnosis. The goal of this project is to predict whether a breast cancer diagnosis is malignant or benign based on various features using machine learning techniques.

## Dataset
The breast cancer dataset used in this project is loaded from a CSV file named breast-cancer.csv. The dataset contains information about various features related to breast cancer tumors.

## Features
The features used for prediction are extracted from the dataset by dropping the 'diagnosis' column, which represents the target variable. The remaining features include information about the characteristics of the tumors.

## Preprocessing
Before training the machine learning model, the dataset is preprocessed:

## Data Loading: 
The dataset is loaded using the Pandas library.
## Data Exploration: 
Basic information about the dataset is displayed, including its shape and summary statistics.
## Handling Missing Values:
Any missing values in the dataset are identified and handled if necessary.
## Data Splitting: 
The dataset is split into training and testing sets using a test size of 20% and a random state of 42.
## Model Training
A Logistic Regression model is used for prediction:

## Model Initialization:
An instance of the LogisticRegression model is created.
## Model Fitting
The model is trained on the training data.
## Training and Testing Accuracy: 
The accuracy of the model on both training and testing data is calculated and displayed.
## Dependencies
NumPy
Pandas
scikit-learn (sklearn)
joblib
# Real-time Face Mesh Detection using OpenCV and MediaPipe
https://github.com/hemanitekwani/ml-nlp_projects/blob/main/facemesh.py

This repository contains code for a real-time face mesh detection application using OpenCV and MediaPipe. The application captures video from the default camera, processes each frame, and detects facial landmarks using the MediaPipe FaceMesh module. Detected landmarks are visualized on the video frame in real-time.

## Usage
To use the face mesh detection application:

## Clone the repository.
Install the required dependencies (OpenCV, MediaPipe).
Run the provided Python script.
Press 'q' to exit the application.
Fake News Detection using Natural Language Processing
This repository contains code for a fake news detection project using NLP techniques. The project classifies news articles as real or fake based on their content. Data preprocessing, TF-IDF vectorization, and training a Logistic Regression model are included.

## Dataset
The project uses a subset of a news dataset loaded from a CSV file named train.csv.

## Data Preprocessing
Data Loading using Pandas.
## Handling Missing Values by filling them.
Feature Creation by combining 'author' and 'title' columns.
Text Preprocessing includes converting to lowercase, tokenizing, stemming, and removing stopwords.
## Feature Extraction
TF-IDF vectorization converts processed text to numerical features.

## Model Training and Evaluation
Data Splitting: Training and testing sets are split.
Model Initialization and Training: Logistic Regression model is trained.
Accuracy Calculation: Model accuracy on training and testing data is calculated.

