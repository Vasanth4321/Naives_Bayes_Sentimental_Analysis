# Naive Bayes Restaurant Review Sentiment Analysis

An end-to-end sentiment analysis project that classifies restaurant reviews as positive or negative using a **Naive** Bayes classifier and an interactive Streamlit web application.

🔗 **Live App:**  
[https://naivesbayessentimentalanalysis-8d955uosdengaiuad2xfkx.streamlit.app/](https://naivesbayessentimentalanalysis-8d955uosdengaiuad2xfkx.streamlit.app/)


## Project Overview

This project focuses on understanding customer opinions about restaurants by analyzing text reviews.  
The pipeline covers data cleaning, text preprocessing, feature extraction, model training, evaluation, and deployment as a user-friendly web app.

Key goals:

- Build a text classification model for restaurant review sentiment.  
- Enable users to input custom reviews and instantly see predicted sentiment.  
- Deploy the model using Streamlit for easy web access.


## Features

- Text preprocessing (lowercasing, removing noise, tokenization, stopword removal, etc.).  
- Feature extraction with techniques like Bag-of-Words or TF-IDF (depending on your implementation).  
- Naive Bayes classifier trained on labeled restaurant review data.  
- Interactive Streamlit UI to:
  - Enter a restaurant review
  - View predicted sentiment (Positive/Negative)
  - See basic model output in real time  


## Tech Stack

- **Programming Language:** Python  
- **Libraries (Core):**  
  - scikit-learn (Naive Bayes, train-test split, metrics)  
  - pandas, numpy (data handling)  
  - nltk or similar (text preprocessing, stopwords)
- **Web Framework:** Streamlit
- **Deployment:** Streamlit Community Cloud


## How It Works

1. **Data Ingestion**  
   - Load restaurant review dataset with text and sentiment labels (e.g., positive/negative).

2. **Preprocessing**  
   - Clean text (remove punctuation, special characters, etc.).  
   - Convert to lowercase, remove stopwords, optionally apply stemming/lemmatization.

3. **Feature Extraction**  
   - Transform text into numerical vectors using CountVectorizer or TfidfVectorizer.

4. **Model Training**  
   - Train a Naive Bayes classifier (e.g., MultinomialNB) on the vectorized reviews.
   - Evaluate using accuracy and other metrics on a test set.

5. **Deployment**  
   - Wrap preprocessing + model prediction into a function.  
   - Expose this via a Streamlit interface for real-time predictions.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Vasanth4321/Naives_Bayes_Sentimental_Analysis.git
cd Naives_Bayes_Sentimental_Analysis
```

### 2. (Optional) Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App Locally
```bash
streamlit run app.py
```

##### Open the URL shown in your terminal (usually http://localhost:8501).

## Usage
- Open the live app or run it locally.
- Enter a restaurant review in the text box.
- Click the predict button to see whether the sentiment is Positive or Negative.

## Example:

> "The ambience was great and the food was delicious!" → Positive

## Author
Venkata Sai Vasanth Neeli
