# ==========================================================
# Sentiment Analysis - Prediction
# ==========================================================

import re
import string
import joblib
import nltk

from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ----------------------------------------------------------
# Load Saved Model
# ----------------------------------------------------------

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

print("Model Loaded Successfully!")

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
# Predict Sentiment
# ----------------------------------------------------------

while True:

    user_input = input("\nEnter a sentence (type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("Thank You!")
        break

    cleaned_text = clean_text(user_input)

    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)[0]

    print("\nPredicted Sentiment:", prediction)