import streamlit as st
import sklearn
import pickle
import string
import re
import nltk
import time
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')

port_stemmer = PorterStemmer()

# Load model & vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Clean text
def clean_text(text):
    text = word_tokenize(text)
    text = " ".join(text)
    text = [ch for ch in text if ch not in string.punctuation]
    text = ''.join(text)
    text = [ch for ch in text if ch not in re.findall(r"[0-9]", text)]
    text = ''.join(text)
    text = [w.lower() for w in text.split() if w.lower() not in set(stopwords.words('english'))]
    text = ' '.join(text)
    text = list(map(lambda x: port_stemmer.stem(x), text.split()))
    return " ".join(text)

# Streamlit Page Config
st.set_page_config(page_title="AI SMS Spam Detector", page_icon="🤖", layout="centered")

# ---- Custom Dark Theme CSS ----
st.markdown("""
    <style>
        .block-container { padding-top: 1rem !important; }
        body {
            background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        h1 { text-align: center; color: #00c6ff; text-shadow: 0 0 15px #00c6ff; letter-spacing: 1px; }
        h3 { color: #aaa; text-align: center; margin-bottom: 1.5rem; }

        section[data-testid="stTextInputRoot"], section[data-testid="stTextAreaRoot"] { background: transparent !important; box-shadow: none !important; }
        div[data-baseweb="base-input"], div[data-baseweb="textarea"] { background: transparent !important; }

        .stTextArea textarea {
            border-radius: 12px; border: 1px solid #00c6ff;
            background-color: rgba(15,15,15,0.8);
            color: white; font-size: 1rem; padding: 0.7rem;
        }

        .stButton button {
            border: none; font-weight: 600; font-size: 1rem;
            border-radius: 12px; transition: 0.3s ease;
        }
        .predict-btn button {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white; box-shadow: 0 0 10px #00c6ff;
        }
        .predict-btn button:hover { box-shadow: 0 0 20px #00c6ff; transform: scale(1.05); }
        .clear-btn button {
            background: linear-gradient(90deg, #ff0844, #ffb199);
            color: white; box-shadow: 0 0 10px #ff0844;
        }
        .clear-btn button:hover { box-shadow: 0 0 20px #ff0844; transform: scale(1.05); }

        .progress-bar {
            height: 25px; border-radius: 20px; color: white;
            text-align: center; font-weight: bold; line-height: 25px;
            transition: width 1.5s ease;
        }
        .footer {
            text-align:center; color: #aaa; font-size: 0.9rem; margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("<h1>🤖 SMS Spam Detection Using Explainable AI</h1>", unsafe_allow_html=True)
st.markdown("<h3>💬 One click to catch every scam</h3>", unsafe_allow_html=True)

# ---- Input Section ----
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
input_sms = st.text_area("📨 Enter your message:", height=150, placeholder="e.g. Congratulations! You won a free iPhone 🎉")

col1, col2 = st.columns(2)
with col1:
    predict_button = st.button("🔎 Predict", key="predict", use_container_width=True)
with col2:
    clear_button = st.button("🧹 Clear", key="clear", use_container_width=True)

if clear_button:
    st.rerun()

# ---- Prediction ----
if predict_button:
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        with st.spinner("🤖 Analyzing your message..."):
            time.sleep(1.5)

        cleaned = clean_text(input_sms)
        vector_input = tfidf.transform([cleaned])
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]

        # Common keywords
        spam_keywords = ["win", "offer", "free", "claim", "urgent", "congratulations", "lottery", "prize", "credit", "money", "buy", "sale"]
        ham_keywords = ["hello", "meeting", "thanks", "please", "regards", "ok", "sure", "see you", "good morning"]

        if result == 1:
            label = "Spam"
            confidence = proba[1] * 100
            bar_color = "linear-gradient(90deg, #ff0844, #ffb199)"
            st.error(f"🚨 **{label.upper()} DETECTED!**")

            matched = [w for w in spam_keywords if w in input_sms.lower()]
            if matched:
                reason = f"Message contains promotional/spam-related keywords: {', '.join(matched)}."
            elif any(x in input_sms.lower() for x in ["http", "www", ".com", "click", "link"]):
                reason = "Message contains links or call-to-action terms typical of spam."
            elif sum(1 for c in input_sms if c.isupper()) > len(input_sms) * 0.4:
                reason = "Message uses excessive uppercase letters, often seen in spam or ads."
            else:
                reason = "Message structure and tone resemble common spam communication patterns."

            st.info(reason)

        else:
            label = "Ham"
            confidence = proba[0] * 100
            bar_color = "linear-gradient(90deg, #56ab2f, #a8e063)"
            st.success(f"✅ **Safe Message: {label}**")

            matched = [w for w in ham_keywords if w in input_sms.lower()]
            if matched:
                reason = f"Message includes regular conversational words: {', '.join(matched)}."
            elif len(input_sms.split()) < 5:
                reason = "Short message with no suspicious structure detected."
            else:
                reason = "Message appears natural and contextually safe."

            st.success(reason)

        # Confidence Bar
        st.markdown(f"""
        <div class="progress-bar" style="width:{int(confidence)}%; background:{bar_color};">
            {int(confidence)}% Confidence
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---- Footer ----
st.markdown('<p class="footer">✨ Built by <b>YASH</b> </p>', unsafe_allow_html=True)
