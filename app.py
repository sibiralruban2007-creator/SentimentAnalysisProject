import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download stopwords
nltk.download('stopwords')

# -----------------------------
# Load Dataset
# -----------------------------
# Replace with your dataset file name
data = pd.read_csv("reviews.csv")

# Example:
# Column names should be:
# review -> text review
# sentiment -> positive / negative

# -----------------------------
# Text Cleaning Function
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning
data['clean_review'] = data['review'].apply(clean_text)

# -----------------------------
# Feature Extraction
# -----------------------------
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data['clean_review'])
y = data['sentiment']

# -----------------------------
# Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Online Product Review Sentiment Analysis")

st.write("Enter a product review to predict sentiment.")

user_input = st.text_area("Enter Review")

if st.button("Predict Sentiment"):

    cleaned_input = clean_text(user_input)

    input_vector = vectorizer.transform([cleaned_input])

    prediction = model.predict(input_vector)

    if prediction[0] == "positive":
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😞")