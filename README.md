# 🧠 Emotion Miscommunication Detection System

## 📌 Overview

The **Emotion Miscommunication Detection System** is a deep learning-based web application that analyzes text input to detect:

* 🎭 Emotions (Happy, Sad, Fear, Angry, etc.)
* 😏 Sarcasm (Yes / No)

This project uses transformer-based models to understand user intent and identify potential miscommunication in text.

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

### ▶️ Step 4: Run the Application

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
5. Results displayed with confidence scores

---

## ⚠️ Known Issues

* Sarcasm detection may over-predict "Yes" for certain negative sentences
* Performance depends on training dataset quality

---

## 🔮 Future Improvements

* Improve sarcasm detection accuracy
* Add more emotion classes
* Deploy as a web application
* Optimize model size for faster inference

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

---

## 📌 Note

Make sure Git LFS is installed before cloning so that model files are downloaded correctly.
