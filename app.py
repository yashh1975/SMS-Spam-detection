import streamlit as st
import pickle
import string
import re
import nltk
import time
import warnings
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ------------------ WARNINGS ------------------
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------ PATH FIX (CRITICAL) ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------ NLTK SETUP (DEPLOYMENT SAFE) ------------------
nltk_data_dir = os.path.join(BASE_DIR, "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_dir)
    nltk.download("punkt", download_dir=nltk_data_dir)

# ------------------ NLP TOOLS ------------------
port_stemmer = PorterStemmer()

# ------------------ LOAD MODEL & VECTORIZER ------------------
tfidf = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

# ------------------ TEXT CLEANING ------------------
def clean_text(text):
    text = word_tokenize(text)
    text = " ".join(text)

    text = [ch for ch in text if ch not in string.punctuation]
    text = ''.join(text)

    text = [ch for ch in text if ch not in re.findall(r"[0-9]", text)]
    text = ''.join(text)

    text = [
        w.lower()
        for w in text.split()
        if w.lower() not in set(stopwords.words("english"))
    ]

    text = " ".join(text)
    text = list(map(lambda x: port_stemmer.stem(x), text.split()))

    return " ".join(text)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI SMS Spam Detector",
    page_icon="🤖",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.block-container { padding-top: 1rem !important; }
body {
    background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Poppins', sans-serif;
}
h1 { text-align: center; color: #00c6ff; text-shadow: 0 0 15px #00c6ff; }
h3 { color: #aaa; text-align: center; }

.stTextArea textarea {
    border-radius: 12px;
    border: 1px solid #00c6ff;
    background-color: rgba(15,15,15,0.8);
    color: white;
}

.stButton button {
    border-radius: 12px;
    font-weight: 600;
}

.progress-bar {
    height: 25px;
    border-radius: 20px;
    color: white;
    text-align: center;
    font-weight: bold;
    line-height: 25px;
}
.footer {
    text-align:center;
    color: #aaa;
    font-size: 0.9rem;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("<h1>🤖 SMS Spam Detection Using Explainable AI</h1>", unsafe_allow_html=True)
st.markdown("<h3>💬 One click to catch every scam</h3>", unsafe_allow_html=True)

# ------------------ INPUT ------------------
input_sms = st.text_area(
    "📨 Enter your message:",
    height=150,
    placeholder="e.g. Congratulations! You won a free iPhone 🎉"
)

col1, col2 = st.columns(2)
with col1:
    predict_button = st.button("🔎 Predict", use_container_width=True)
with col2:
    clear_button = st.button("🧹 Clear", use_container_width=True)

if clear_button:
    st.rerun()

# ------------------ PREDICTION ------------------
if predict_button:
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        with st.spinner("🤖 Analyzing your message..."):
            time.sleep(1.2)

        cleaned = clean_text(input_sms)
        vector_input = tfidf.transform([cleaned])
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]

        spam_keywords = [
            "win", "offer", "free", "claim", "urgent",
            "congratulations", "lottery", "prize",
            "credit", "money", "buy", "sale"
        ]

        ham_keywords = [
            "hello", "meeting", "thanks", "please",
            "regards", "ok", "sure", "see you"
        ]

        if result == 1:
            confidence = proba[1] * 100
            st.error("🚨 **SPAM DETECTED!**")

            matched = [w for w in spam_keywords if w in input_sms.lower()]
            if matched:
                reason = f"Contains spam keywords: {', '.join(matched)}."
            elif any(x in input_sms.lower() for x in ["http", "www", ".com", "link"]):
                reason = "Contains suspicious links or call-to-action phrases."
            else:
                reason = "Pattern and tone resemble known spam messages."

            st.info(reason)
            bar_color = "linear-gradient(90deg, #ff0844, #ffb199)"

        else:
            confidence = proba[0] * 100
            st.success("✅ **SAFE MESSAGE (HAM)**")

            matched = [w for w in ham_keywords if w in input_sms.lower()]
            if matched:
                reason = f"Contains conversational words: {', '.join(matched)}."
            else:
                reason = "Message structure appears natural and safe."

            st.success(reason)
            bar_color = "linear-gradient(90deg, #56ab2f, #a8e063)"

        st.markdown(f"""
        <div class="progress-bar" style="width:{int(confidence)}%; background:{bar_color};">
            {int(confidence)}% Confidence
        </div>
        """, unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown('<p class="footer">✨ Built by <b>YASH</b></p>', unsafe_allow_html=True)
