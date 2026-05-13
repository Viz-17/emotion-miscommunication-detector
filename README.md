# 🧠 Emotion Miscommunication Detection System

## 📌 Overview

The **Emotion Miscommunication Detection System** is a deep learning-based web application that analyzes text input to detect:

* 🎭 Emotions (Happy, Sad, Fear, Angry, etc.)
* 😏 Sarcasm (Yes / No)

This project uses transformer-based models to understand user intent and identify potential miscommunication in text.

🆕 **Latest Update:**
Now enhanced with **AI-powered message rewriting using LLM (LLaMA 3.1 via Groq)** to automatically correct miscommunication and suggest better phrasing.

---

## 🚀 Features

* 🔍 Emotion Classification using NLP
* 🤖 Sarcasm Detection
* 🧠 Multiple Model Support:

  * BERT
  * DistilBERT
  * RoBERTa
* 📊 Confidence Scores for predictions
* 🌐 Interactive UI using Streamlit
* ⚡ Real-time text analysis

🆕 **New Features Added:**

* ✨ **AI-Powered Message Rewriter (LLM Integration)**

  * Automatically rewrites messages when miscommunication is detected
  * Converts sarcastic or unclear text into **clear, professional communication**

* 🧾 **Intent vs Perception Analysis**

  * Compares user-selected emotion with AI-predicted emotion

* 🎯 **Smart Sarcasm Handling**

  * Adjusts emotion output when sarcasm is detected

* 🧠 **Best Model Selection**

  * Dynamically selects the most confident prediction among BERT, DistilBERT, and RoBERTa

---

## 🏗️ Tech Stack

### 💻 Backend

* Python
* PyTorch
* Transformers (HuggingFace)

### 📊 Machine Learning

* BERT
* DistilBERT
* RoBERTa

### 🌐 Frontend

* Streamlit

### 📦 Others

* NumPy
* Pandas
* Scikit-learn

🆕 **Additional Technologies:**

* Groq API (LLaMA 3.1)
* Prompt Engineering for controlled text generation

---

## ⚙️ Installation & Setup

### 🔑 Prerequisites

Make sure you have:

* Python 3.8+
* pip installed

---

### 📥 Step 1: Clone the Repository

```
git clone https://github.com/Viz-17/emotion-miscommunication-detector.git
cd emotion-miscommunication-detector
```

---

### 📦 Step 2: Install Dependencies

```
pip install -r requirements.txt
```

---

### ⚠️ Step 3: Install Git LFS (IMPORTANT)

This project uses large model files (.safetensors).

```
git lfs install
```

---

### 🔐 Step 4: Setup Environment Variables (NEW)

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_api_key_here
```

⚠️ **Important:**
Never hardcode API keys in the source code.

---

### ▶️ Step 5: Run the Application

```
streamlit run app.py
```

---

## 🧪 How It Works

1. User inputs text
2. Text is tokenized using pretrained tokenizer
3. Passed through transformer models
4. Model predicts:

   * Emotion label
   * Sarcasm detection
5. Best model is selected based on confidence
6. Sarcasm adjustment logic is applied
7. Intent vs perception mismatch is evaluated

🆕 **If mismatch detected:**

8. LLM (LLaMA 3.1 via Groq) generates:

   * A **clear**
   * **Professional**
   * **Non-sarcastic rewritten message**

9. Results displayed with confidence scores and suggestions

---

## ⚠️ Known Issues

* Sarcasm detection may over-predict "Yes" for certain negative sentences
* Performance depends on training dataset quality

🆕 Additional Notes:

* LLM output may vary slightly due to probabilistic generation
* Requires internet connection for AI rewriting feature

---

## 🔮 Future Improvements

* Improve sarcasm detection accuracy
* Add more emotion classes
* Deploy as a web application
* Optimize model size for faster inference

🆕 Planned Enhancements:

* Chat-style conversational UI
* Multi-language support
* Model explainability (attention visualization)
* Deployment on Streamlit Cloud / Render
* User feedback loop for model improvement

---

## 🤝 Contribution

Feel free to fork and improve the project.

---

## 👨‍💻 Author

**Vishwa**

---

## ⭐ Acknowledgement

* HuggingFace Transformers
* Streamlit
* Open-source ML community

🆕 Additional Credits:

* Groq (LLaMA 3.1 API)

---

## 📌 Note

Make sure Git LFS is installed before cloning so that model files are downloaded correctly.

⚠️ Ensure `.env` file is properly configured for LLM features to work.
