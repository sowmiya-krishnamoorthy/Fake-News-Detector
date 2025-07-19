import streamlit as st
import joblib
import os

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown("""
    <style>
            body {
               background-color: angular-gradient(
            from 180deg at centre,
            5AB373 10% 100% ,
            A84646 100% 100% );
             }
            
            @import url('https://fonts.googleapis.com/css2?family=Piazzolla&display=swap')

            html, body, [class*="css"] {
        font-family: 'Piazzolla', serif !important;
            }
 
            
        .result-box {
            padding: 1rem;
            border-radius: 10px;
            font-size: 1.5rem;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .real {
            background-color: rgba(189, 249, 206, 1);
            color: rgba(46, 217, 77, 1);
        }
        .fake {
            background-color: rgba(254, 183, 183, 1);
            color: rgba(255, 0, 0, 1) ;
        }
        .stButton>button {
            font-size: 1.1rem;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detector")
st.markdown("Enter news content below and click **Predict** to check if it's Real or Fake.")

# Load model and vectorizer
if os.path.exists("model/fake_news_model.pkl") and os.path.exists("model/tfidf_vectorizer.pkl"):
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
else:
    st.error("Model files not found. Please train your model and save it first.")
    st.stop()

input_text = st.text_area("📝 Paste any news article here:", height=200)

if st.button("🔍 Predict"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        transformed = vectorizer.transform([input_text])
        prediction = model.predict(transformed)[0]
        
        if prediction == 1:
            # Fake news
            label_html = '<div class="result-box fake">🟥 FAKE NEWS</div>'
            st.markdown(label_html, unsafe_allow_html=True)
            st.markdown("""
                <style>
                body {
                    background-color: #ffe6e6;
                }
                .stButton>button {
                    background-color: #cc0000;
                    color: white;
                }
                </style>
            """, unsafe_allow_html=True)
        else:
            # Real news
            label_html = '<div class="result-box real">🟩 REAL NEWS</div>'
            st.markdown(label_html, unsafe_allow_html=True)
            st.markdown("""
                <style>
                body {
                    background-color: #e6ffed;
                }
                .stButton>button {
                    background-color: #007a33;
                    color: white;
                }
                </style>
            """, unsafe_allow_html=True)

