import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
import streamlit as st
import pickle

def clean(doc): 
    regex = "[^a-zA-Z.]"
    doc = re.sub(regex, " ", doc)
    doc = doc.lower()
    tokens = nltk.word_tokenize(doc)

    stop_words = set(stopwords.words('english'))

    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return " ".join(lemmatized_tokens)



# Load the trained model
with open("Naives_Bayes_Sentimental_Analysis.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# App title
st.title("Review Classification")

st.write("Enter a review below to check whether it is **Positive** or **Negative** Review.")

# Text input
review_text = st.text_area("Review Text", height=150)

# Predict button
if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("Please enter a review to predict.")
    else:
        review_text = clean(review_text)
        vectorized_review = vectorizer.transform([review_text])
        prediction = model.predict(vectorized_review)[0]

        if prediction == 0:
            st.error("🚨 Negative Review")
        else:
            st.success("✅ Positive Review")
