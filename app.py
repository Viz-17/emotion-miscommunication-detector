import streamlit as st
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

st.set_page_config(page_title="Miscommunication Detector", page_icon="image.png", layout="wide")

# ==============================
# ✅ Load sarcasm model
# ==============================
@st.cache_resource
def load_sarcasm():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-irony"
    )

sarcasm_detector = load_sarcasm()

# ==============================
# ✅ Load BERT
# ==============================
@st.cache_resource
def load_bert():
    path = "bert_model"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

bert_tokenizer, bert_model = load_bert()

# ==============================
# ✅ Load DistilBERT
# ==============================
@st.cache_resource
def load_distilbert():
    path = "distilbert_model"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

distil_tokenizer, distil_model = load_distilbert()

# ==============================
# ✅ Load RoBERTa
# ==============================
@st.cache_resource
def load_roberta():
    path = "roberta1_model"   # your saved folder
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

roberta_tokenizer, roberta_model = load_roberta()

# ==============================
# ✅ Label mapping
# ==============================
label_map = {
    0: "Angry 😡",
    1: "Fear 😨",
    2: "Happy 😄",
    3: "Love ❤️",
    4: "Neutral 😐",
    5: "Sad 😢"
}

# ==============================
# UI Styles & Sidebar
# ==============================
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #ff4b4b;
        color: white;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("image.png", width=200)
    st.title("About the App")
    st.info("Multiple AI models analyze your text for true emotion, and a dedicated sarcasm detector ensures your intent isn't misunderstood.")
    
    st.markdown("### 🎯 The Problem:")
    st.write("Digital text lacks tone, body language, and facial expressions. This often leads to severe misinterpretations in corporate emails, customer service chats, and social media.")
    
    st.markdown("### 💡 The Solution:")
    st.write("This tool acts as a communication safeguard. By running text through emotion and irony detection, it flags messages that are highly likely to be misunderstood by the receiver before they are sent.")

    st.markdown("### 📖 How to use this tool:")
    st.markdown("1. **Draft your message** in the text box exactly as you would send it to a colleague or friend.")
    st.markdown("2. **Select your true intent** from the dropdown menu (how you actually feel).")
    st.markdown("3. **Analyze** to see how an objective reader (the AI) will likely interpret your tone, and check if hidden sarcasm is overriding your message.")

    st.markdown("---")
    st.markdown("### 💡 Tips:")
    st.markdown("Try entering sarcastic sentences and watch the model adjust the emotional meaning!")

st.markdown('<p class="main-header">💬 Emotion Miscommunication Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze intended vs predicted emotion with advanced sarcasm awareness</p>', unsafe_allow_html=True)

# Main input container
col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area("📝 Enter your message:", height=150, placeholder="e.g., Oh great, another meeting. Just what I wanted...")

with col2:
    intended_emotion = st.selectbox(
        "🎯 Select intended emotion",
        ["Happy 😄", "Sad 😢", "Angry 😡", "Neutral 😐",  "Fear 😨"]
    )
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("🚀 Analyze Emotion")

# ==============================
# Helpers
# ==============================
def normalize_emotion(emotion):
    return emotion.split()[0].lower()

def adjust_for_sarcasm(emotion, is_sarcastic):
    base = normalize_emotion(emotion)

    if not is_sarcastic:
        return emotion

    if base in ["happy", "love"]:
        return "Angry 😡"
    elif base == "neutral":
        return "Sad 😢"
    else:
        return emotion

# ==============================
# Analyze
# ==============================
if analyze_button:

    if text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze!")

    else:
        with st.spinner("🤖 AI is analyzing your message..."):
            time.sleep(0.5)

        # ==============================
        # 🔹 BERT Prediction
        # ==============================
        inputs_bert = bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs_bert = bert_model(**inputs_bert)
            probs_bert = torch.nn.functional.softmax(outputs_bert.logits, dim=1)

            pred_bert = torch.argmax(probs_bert).item()
            conf_bert = probs_bert[0][pred_bert].item()

        bert_emotion = label_map[pred_bert]

        # ==============================
        # 🔹 DistilBERT Prediction
        # ==============================
        inputs_distil = distil_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs_distil = distil_model(**inputs_distil)
            probs_distil = torch.nn.functional.softmax(outputs_distil.logits, dim=1)

            pred_distil = torch.argmax(probs_distil).item()
            conf_distil = probs_distil[0][pred_distil].item()

        distil_emotion = label_map[pred_distil]

        # ==============================
        # 🔹 RoBERTa Prediction
        # ==============================
        inputs_roberta = roberta_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs_roberta = roberta_model(**inputs_roberta)
            probs_roberta = torch.nn.functional.softmax(outputs_roberta.logits, dim=1)

            pred_roberta = torch.argmax(probs_roberta).item()
            conf_roberta = probs_roberta[0][pred_roberta].item()

        roberta_emotion = label_map[pred_roberta]

        
        # ==============================
        # 🏆 Select Best Model (3-way)
        # ==============================
        model_results = [
            ("BERT", bert_emotion, conf_bert),
            ("DistilBERT", distil_emotion, conf_distil),
            ("RoBERTa", roberta_emotion, conf_roberta)
        ]

        best_model, final_pred, final_conf = max(model_results, key=lambda x: x[2])

        # ==============================
        # 😏 Sarcasm Detection
        # ==============================
        sarcasm_result = sarcasm_detector(text)
        sarcasm_label = sarcasm_result[0]['label'].lower()
        sarcasm_score = sarcasm_result[0]['score']

        #is_sarcastic = "irony" in sarcasm_label
        simple_text = text.lower()

        if len(simple_text.split()) < 4:
            is_sarcastic = False
        if "irony" in sarcasm_label and sarcasm_score > 0.95:
            is_sarcastic = True
        else:
            is_sarcastic = False

        # ==============================
        # 🔥 Final Emotion
        # ==============================
        final_emotion = adjust_for_sarcasm(final_pred, is_sarcastic)

        mismatch = normalize_emotion(intended_emotion) != normalize_emotion(final_emotion)

        # ==============================
        # 📊 UI OUTPUT (Tabs for better organization)
        # ==============================
        st.divider()
        
        # Group the results using Tabs to make it look clean
        tab1, tab2, tab3 = st.tabs(["🎯 Summary Result", "🔍 Deep Analysis", "🤖 Model Confidence"])
        
        with tab1:
            st.subheader("Results Overview")
            
            # Status Banner
            if mismatch:
                st.error("### ⚠️ Miscommunication Detected!\nThe emotion perceived by the AI differs from your intention.")
            else:
                st.success("### ✅ Clear Communication!\nThe tone matches your intended emotion perfectly.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Core Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Your Intent", intended_emotion)
            with m2:
                st.metric("AI's Perception", final_emotion, delta="Adjusted" if is_sarcastic else "")
            with m3:
                st.metric("Sarcasm Detected", "Yes 😏" if is_sarcastic else "No")
            with m4:
                st.metric("Mismatch", "Yes ⚠️" if mismatch else "No ✅")
                
            st.markdown("---")
            if is_sarcastic:
                st.info("**Why this happened:** Sarcasm was detected, which flipped the emotional meaning of the sentence.")
            elif mismatch:
                st.warning("**Why this happened:** The predicted emotion differs from your intended meaning. Try rephrasing your sentence to better reflect your intent.")
            else:
                st.success("**Why this happened:** The predicted emotion perfectly aligns with your intent.")

        with tab2:
            st.subheader("Sarcasm Analysis")
            st.write(f"The sarcasm detection model analyzed the text and found a **{sarcasm_score * 100:.2f}%** probability of it being sarcastic/ironic.")
            st.progress(sarcasm_score)
            
            st.markdown("---")
            st.subheader("Emotion Transformation")
            st.write(f"1. **Dominant Raw Emotion:** {final_pred} (Confidence: {final_conf*100:.1f}%)")
            st.write(f"2. **Sarcasm Identified:** {'Yes' if is_sarcastic else 'No'}")
            st.write(f"3. **Final Adjusted Emotion:** {final_emotion}")
            
            with st.expander("💡 Suggestions"):
                st.write("Try rephrasing your sentence to better reflect your intent. For example, explicitly stating feelings might override subtle sarcasm rules.")

        with tab3:
            st.subheader("Model Confidence Comparison")
            st.write(f"Among the three models evaluated, **{best_model}** exhibited the highest confidence score.")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**BERT**")
                st.progress(conf_bert)
                st.caption(f"{conf_bert*100:.2f}%")

            with c2:
                st.write("**DistilBERT**")
                st.progress(conf_distil)
                st.caption(f"{conf_distil*100:.2f}%")
            
            with c3:
                st.write("**RoBERTa**")
                st.progress(conf_roberta)
                st.caption(f"{conf_roberta*100:.2f}%")