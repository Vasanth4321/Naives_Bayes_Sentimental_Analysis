# 🏛️ Naive Bayes Restaurant Review Sentiment Analysis

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://naivesbayessentimentalanalysis-8d955uosdengaiuad2xfkx.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-NaiveBayes-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge)](https://www.nltk.org/)

An end-to-end **NLP sentiment analysis** project that classifies restaurant reviews as **Positive** or **Negative** using a Naive Bayes classifier, NLTK text preprocessing, and a live Streamlit web application.

## 🚀 Live App

👉 **[Click here to try the app](https://naivesbayessentimentalanalysis-8d955uosdengaiuad2xfkx.streamlit.app/)**

---

## 📌 Project Overview

This project focuses on understanding customer opinions about restaurants by analyzing text reviews. The full pipeline covers data cleaning, text preprocessing, feature extraction, model training, evaluation, and deployment as a user-friendly web app.

**Key Goals:**
- Build a text classification model for restaurant review sentiment.
- Enable users to input custom reviews and instantly see predicted sentiment.
- Deploy the model using Streamlit for easy web access.

---

## ✨ Features

- Text preprocessing using NLTK (lowercasing, regex cleaning, tokenization, stopword removal, lemmatization)
- Feature extraction using **CountVectorizer** (Bag-of-Words)
- **Multinomial Naive Bayes** classifier trained on labeled restaurant review data
- Saved model and vectorizer as `.pkl` files for fast inference
- Interactive **Streamlit UI** to:
  - Enter any restaurant review
  - Instantly see **Positive** ✅ or **Negative** 🚨 prediction

---

## 🧠 How It Works

1. **Input** — User enters a restaurant review in the text box
2. **Preprocessing** — Text is cleaned using regex, lowercased, tokenized, stopwords removed, and lemmatized
3. **Vectorization** — Cleaned text is transformed using the saved `CountVectorizer`
4. **Prediction** — The trained Naive Bayes model predicts Positive (1) or Negative (0)
5. **Output** — Result displayed instantly on screen

**Example:**
> "The food was amazing and the service was great!" → ✅ Positive Review

> "Worst experience ever, food was cold and staff was rude." → 🚨 Negative Review

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python |
| **NLP** | NLTK (tokenization, stopwords, lemmatization) |
| **ML Algorithm** | Naive Bayes (MultinomialNB) |
| **Feature Extraction** | CountVectorizer (scikit-learn) |
| **Libraries** | scikit-learn, pandas, numpy, pickle |
| **Web Framework** | Streamlit |
| **Deployment** | Streamlit Community Cloud |

---

## 📁 Project Structure

```
Naives_Bayes_Sentimental_Analysis/
│
├── streamlit.py                        # Main Streamlit app
├── Naives_Bayes_Sentimental_Analysis.pkl  # Trained Naive Bayes model
├── vectorizer.pkl                      # Saved CountVectorizer
├── Naives_Bayes_Senti...               # Jupyter notebook (EDA + training)
├── output.csv                          # Dataset
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## ⚙️ Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vasanth4321/Naives_Bayes_Sentimental_Analysis.git
   cd Naives_Bayes_Sentimental_Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run streamlit.py
   ```

4. Open your browser at `http://localhost:8501`

---

## 🙋 Author

**Venkata Sai Vasanth Neeli**  
[![GitHub](https://img.shields.io/badge/GitHub-Vasanth4321-black?style=flat&logo=github)](https://github.com/Vasanth4321)
