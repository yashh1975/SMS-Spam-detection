# 📩 SMS Spam Detection with Explainable AI

🔗 **Live Application**  
👉https://ai-sms-spam-detector.streamlit.app/

---

## Problem Statement
The increasing volume of unsolicited and fraudulent SMS messages creates security, privacy, and usability concerns for users. Traditional rule-based or manual filtering methods are inefficient and unreliable.

This project addresses the problem by using **machine learning with explainable AI techniques** to automatically detect spam messages while also providing insights into why a message is classified as spam or not spam.

---

## Project Presentation
The complete project presentation (PPT) explaining the problem statement, methodology, architecture, and results is included in this repository for reference.

📄 **Presentation File:**  

`docs/SMS_Spam_Detection_Presentation.pptx`

---

## Key Features
- ✅ Real-time SMS spam classification  
- 🧠 NLP-based text preprocessing and TF-IDF vectorization  
- 🔍 Explainable AI through keyword influence analysis  
- 📊 Model performance evaluation and visualization  
- ⚡ Fast predictions using a pre-trained machine learning model  
- 🌐 Interactive and user-friendly web interface using Streamlit  

---

## Dataset
The project uses the **SMS Spam Collection Dataset** from the **UCI Machine Learning Repository**.

- **Total messages:** 5,574  
- **Labels:**
  - `1` → Spam  
  - `0` → Ham (Not Spam)  
- **Columns:**
  - `label` – Message category  
  - `text` – SMS content  

📌 **Dataset Source:**  
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

---

## Technical Architecture

### Text Processing
- Text cleaning and normalization  
- Tokenization  
- Stop-word removal and stemming  
- TF-IDF (Term Frequency–Inverse Document Frequency) vectorization  

### Machine Learning
- Supervised classification model (Naïve Bayes)  
- Trained on labeled SMS data  
- Model and vectorizer serialized using `pickle` for reuse  

### Explainable AI
- Identification of influential words contributing to predictions  
- Transparency in spam and non-spam classification decisions  

### Deployment
- Streamlit Community Cloud  
- Model and vectorizer loaded at runtime  
- Stateless and fast inference  

---

## Technology Stack

### Programming Language
- Python 3.x  

### Libraries & Tools
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  
- Matplotlib  

---

## Installation & Local Execution

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/SMS-Spam-Detection.git
cd SMS-Spam-Detection
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Application
```bash
streamlit run app.py
```

---

## Model Performance
The trained model was evaluated using standard classification metrics.

| Metric     | Score   |
|-----------|---------|
| Accuracy  | 97.22%  |
| Precision | 100%    |
| Recall    | 76.19%  |
| F1-Score  | 88.39%  |

These results demonstrate strong spam detection capability with high precision and reliable explainability.

---

## Use Cases
- Academic mini-project or final-year project  
- Demonstration of Explainable AI in NLP applications  
- Resume and portfolio project  
- Foundation for SMS, email, or message filtering systems  

---

## Future Enhancements
- Multilingual spam detection  
- Deep learning–based models with explainability  
- Email and messaging platform integration  
- Database support for message history  
- REST API for external system integration  

---

## License
This project is developed for **educational and learning purposes**. Public datasets are used in accordance with their respective licenses.

---

## Author
**Yash**

---
