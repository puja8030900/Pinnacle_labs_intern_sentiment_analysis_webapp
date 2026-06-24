# ==========================================================
# Sentiment Analysis Web App using Flask
# ==========================================================

from flask import Flask, render_template, request
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords

# Download stopwords (first time only)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ----------------------------------------------------------
# Load Saved Model
# ----------------------------------------------------------

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ----------------------------------------------------------
# Create Flask App
# ----------------------------------------------------------

app = Flask(__name__)

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

    return " ".join(words)

# ----------------------------------------------------------
# Home Page
# ----------------------------------------------------------

# ----------------------------------------------------------
# Home Page
# ----------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")
# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

@app.route('/predict', methods=['POST'])

def predict():

    user_text = request.form["message"]

    cleaned = clean_text(user_text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        message=user_text
    )

# ----------------------------------------------------------
# Run App
# ----------------------------------------------------------


if __name__ == "__main__":
    app.run(debug=True)
