# ==========================================================
# Sentiment Analysis Project
# Author : Sudha Rani
# ==========================================================

# Import Libraries

import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ----------------------------------------------------------
# Download Stopwords
# ----------------------------------------------------------

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

print("Loading Dataset...")

data = pd.read_csv("dataset/train.csv", encoding="latin-1")

print("Dataset Loaded Successfully")

print(data.head())


# ----------------------------------------------------------
# Check Dataset
# ----------------------------------------------------------

print("\nDataset Shape")

print(data.shape)

print("\nColumns")

print(data.columns)

print("\nMissing Values")

print(data.isnull().sum())


# ----------------------------------------------------------
# Select Text and Sentiment Columns
# ----------------------------------------------------------

# Change these column names if your dataset is different

TEXT_COLUMN = "text"

TARGET_COLUMN = "sentiment"


# ----------------------------------------------------------
# Text Cleaning Function
# ----------------------------------------------------------

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()

    words = [word for word in words if word not in stop_words]

    text = " ".join(words)

    return text


# ----------------------------------------------------------
# Apply Cleaning
# ----------------------------------------------------------

print("\nCleaning Text...")

data["clean_text"] = data[TEXT_COLUMN].apply(clean_text)

print("Cleaning Completed")


# ----------------------------------------------------------
# Features and Labels
# ----------------------------------------------------------

X = data["clean_text"]

y = data[TARGET_COLUMN]


# ----------------------------------------------------------
# Convert Text into Numbers
# ----------------------------------------------------------

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)


# ----------------------------------------------------------
# Train Test Split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)


# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

print("\nTraining Model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Training Completed")


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

prediction = model.predict(X_test)


# ----------------------------------------------------------
# Accuracy
# ----------------------------------------------------------

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy")

print(round(accuracy * 100, 2), "%")


# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report")

print(classification_report(y_test, prediction))


# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

print("\nConfusion Matrix")

print(confusion_matrix(y_test, prediction))


# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

joblib.dump(model, "models/model.pkl")

joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel Saved Successfully")

print("vectorizer.pkl Saved")

print("model.pkl Saved")