import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_fake_news.csv")

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Drop rows with missing 'clean_text'
df = df.dropna(subset=["clean_text"])

# Features and labels
x = df["clean_text"]
y = df["label"]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_vec = vectorizer.fit_transform(x)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Save the model and vectorizer
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")