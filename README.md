
# 📰 Fake News Detector using NLP 🧠

A smart and lightweight web application that detects whether a news article is **Real** or **Fake** using advanced Natural Language Processing (NLP) and Machine Learning techniques. This app offers both **text input** and **image upload** (OCR) options, making it easy to verify the authenticity of news from articles or screenshots. Built with **Streamlit**, it features a clean interface, instant predictions, and easy deployment via **ngrok**.


---

## 📌 Features

- Built using **Python**, **Scikit-learn**, and **NLP**
- Web interface with **Streamlit**
- Accepts **text input** from the user
- NEW: Supports **image upload** of news screenshots (using OCR)
- Publicly shareable via **ngrok**
- Predicts if the news is Fake or Real

---

## 🧠 Tech Stack

- **Python**
- **Scikit-learn**
- **Pandas / Numpy**
- **Natural Language Toolkit (NLTK)**
- **Streamlit**
- **Ngrok**


---

## 📁 Project Structure

Fake-News-Detector/
├── app.py # Streamlit web app
├── model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Required packages
└── README.md # Project documentation


---

## 🖥️ How to Run the Project

### 🔧 Step 1: Clone the Repository

git clone https://github.com/sowmiya-krishnamoorthy/fake-news-detector.git
cd fake-news-detector

## 🔧 Step 2: Create a Virtual Environment (Optional but Recommended)

python -m venv venv

source venv/bin/activate       # On Linux/macOS

venv\Scripts\activate          # On Windows

## 🔧 Step 3: Install the Dependencies
pip install -r requirements.txt

## 🔧 Step 4: Run the Application
streamlit run app.py

---


## ✅ How It Works

1.Input a news article into the text box.
2.The app processes the text using a TF-IDF vectorizer.
3.A trained Logistic Regression model analyzes the text.
4.The result is displayed as either Real or Fake.


---
