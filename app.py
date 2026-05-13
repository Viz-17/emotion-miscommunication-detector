import streamlit as st
import time
import torch
import os
from groq import Groq
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline,AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

from groq import Groq

client = Groq(api_key=groq_api_key)

st.set_page_config(page_title="Miscommunication Detector", page_icon="image.png", layout="wide")

# os.environ["GROQ_API_KEY"] = ""
groq_client = Groq()

# sarcasm model
@st.cache_resource
def load_sarcasm():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-irony"
    )

sarcasm_detector = load_sarcasm()

# BERT
@st.cache_resource
def load_bert():
    path = "bert_model"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

bert_tokenizer, bert_model = load_bert()

# DistilBERT
@st.cache_resource
def load_distilbert():
    path = "distilbert_model"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

distil_tokenizer, distil_model = load_distilbert()

# RoBERTa
@st.cache_resource
def load_roberta():
    path = "roberta1_model"   
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

roberta_tokenizer, roberta_model = load_roberta()

# T5 for Generative Correction 

# @st.cache_resource
# def load_corrector():
#     # flan-t5-small is faster for local demos; base is more accurate
# #     return pipeline("text2text-generation", model="google/flan-t5-base")

# # corrector = load_corrector()
#     tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
#     model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
#     return tokenizer, model

# tokenizer_t5, model_t5 = load_corrector()

# Label mapping
label_map = {
    0: "Angry 😡",
    1: "Fear 😨",
    2: "Happy 😄",
    3: "Love ❤️",
    4: "Neutral 😐",
    5: "Sad 😢"
}


# UI Styles & Sidebar
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
    st.info("Multiple transformer-based AI models analyze your text for true emotion, and a dedicated sarcasm detector ensures your intent isn't misunderstood.")
    
    st.markdown("### 🎯 The Problem:")
    st.write("Digital text lacks tone, body language, and facial expressions. This often leads to severe misinterpretations in corporate emails, customer service chats, and social media.")
    
    st.markdown("### 💡 The Solution:")
    st.write("This tool acts as a communication safeguard. By running text through emotion and irony detection, it flags messages that are highly likely to be misunderstood by the receiver before they are sent.")

    st.markdown("### 📖 How to use this tool:")

    st.markdown("""
    1. **Enter your message**  
    Paste a received message *or* draft a message exactly as you would send it.

    2. **(Optional) Select your true intent**  
    If you're drafting a message, choose how you actually feel from the dropdown.

    3. **Click Analyze**  
    The system processes your message using AI models.

    4. **View the results**  
    You will see:
    - The **detected emotion** (Happy, Sad, Angry, Neutral, etc.)
    - Whether the message contains **sarcasm**
    - The **interpreted tone** behind the message  
    - (If intent selected) comparison between **your intent vs perceived tone**
    """)

    st.markdown("## 📈 What the Model Analyzes")

    st.markdown("""
    The system evaluates:
    - **Emotion Classification** (Happy, Sad, Angry, Fear, Neutral)
    - **Sarcasm Detection**
    - **Intent vs Perception Gap**

    These combined insights help detect miscommunication.
    """)

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


if analyze_button:

    if text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze!")

    else:
        with st.spinner("🤖 AI is analyzing your message..."):
            time.sleep(0.5)

        
        # BERT Prediction
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

        # DistilBERT Prediction
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

        # RoBERTa Prediction
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

        
        # Selecting Best Model (3-way)
        model_results = [
            ("BERT", bert_emotion, conf_bert),
            ("DistilBERT", distil_emotion, conf_distil),
            ("RoBERTa", roberta_emotion, conf_roberta)
        ]

        best_model, final_pred, final_conf = max(model_results, key=lambda x: x[2])

        # Sarcasm Detection
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

        # Final Emotion
        final_emotion = adjust_for_sarcasm(final_pred, is_sarcastic)

        mismatch = normalize_emotion(intended_emotion) != normalize_emotion(final_emotion)

        # UI OUTPUT 
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["🎯 Summary Result", "🔍 Deep Analysis", "🤖 Model Confidence"])
        
        with tab1:
            st.subheader("Results Overview")
            
            # if mismatch:
            #     st.error("### ⚠️ Miscommunication Detected!\nThe emotion perceived by the AI differs from your intention.")
            # else:
            #     st.success("### ✅ Clear Communication!\nThe tone matches your intended emotion perfectly.")
            
            # st.markdown("<br>", unsafe_allow_html=True)
            
            # # Core Metrics
            # m1, m2, m3, m4 = st.columns(4)
            # with m1:
            #     st.metric("Your Intent", intended_emotion)
            # with m2:
            #     st.metric("AI's Perception", final_emotion, delta="Adjusted" if is_sarcastic else "")
            # with m3:
            #     st.metric("Sarcasm Detected", "Yes 😏" if is_sarcastic else "No")
            # with m4:
            #     st.metric("Mismatch", "Yes ⚠️" if mismatch else "No ✅")
            
            if mismatch:
                st.error("### ⚠️ Miscommunication Detected!")
                st.write("The emotion perceived by the AI differs from your intention.")
                
                st.markdown("---")
                st.subheader("🤖 AI Suggested Revision")
                with st.spinner("Generating a polite version..."):
                    try:
                            #prompt = f"Rewrite the following sentence in a highly polite, professional, and {intended_emotion} tone without changing its core factual meaning. Only return the revised sentence, nothing else. Sentence: {text}"
                            if is_sarcastic:
                                prompt = (
                                    f"The following sentence is sarcastic and may not reflect its true meaning. "
                                    f"First, interpret the actual intended meaning behind the sentence. "
                                    f"Then rewrite it into a clear, polite, short, direct, and professional sentence suitable for workplace communication. "
                                    f"Remove sarcasm and ambiguity. "
                                    f"- Use neutral, objective corporate tone\n"
                                    f"- Preserve exact meaning\n"
                                    f"Only return the revised sentence.\n\nSentence: \"{text}\""
                                    # f"Rewrite the following sentence into a short, direct, and clear sentence.\n\n"
                                    # f"CRITICAL RULES:\n"
                                    # f"- Maximum 12 words\n"
                                    # f"- Preserve exact meaning\n"
                                    # f"- Do not expand or generalize\n"
                                    # f"- Do not use corporate language\n"
                                    # f"- Do not use 'we'\n"
                                    # f"- Keep it simple and human\n"
                                    # f"- Output only one sentence\n\n"
                                    # f"Sentence: \"{text}\""
                                )
                            else:
                                prompt = (
                                    f"Rewrite the following sentence into a polite, professional, and emotionally appropriate tone "
                                    f"for workplace communication. Preserve the original meaning. "
                                    f"Only return the revised sentence.\n\nSentence: \"{text}\""
                                )
                            chat_completion = groq_client.chat.completions.create(
                                messages=[
                                    {
                                        "role": "system", 
                                        #"content": "You are a strict, professional corporate communications assistant. You must ONLY output the revised sentence. Never include conversational filler, greetings, or introductory text."
                                        "content":"You are a strict, professional corporate communications assistant. Always produce a direct sentence. Never expand, never generalize,  never use 'we'."
                                    },
                                    {
                                        "role": "user", 
                                        "content": prompt
                                    }
                                ],
                                model="llama-3.1-8b-instant",
                                temperature=0.2, 
                            )
                            
                            suggestion = chat_completion.choices[0].message.content.strip()
                            suggestion = suggestion.strip().replace('"', '')
                            st.success(f"**Try sending this instead:** \n\n{suggestion}")
                            
                    except Exception as e:
                        st.error(f"Generative API Error. Please check your network or Groq API Key. Details: {e}")
                    #prompt = f"Rewrite the following sentence in a polite and {intended_emotion} tone without changing its meaning: {text}"
                    #prompt = f"Rewrite this sentence to be highly polite, professional, constructive, and a {intended_emotion} tone without changing its meaning in a workplace setting: '{text}'"
                    # prompt = f"""Convert the following rude sentence into a polite and professional workplace sentence.

                    # Rude: This work is garbage, do it again.
                    # Polite: Could you please review this work and make the necessary improvements?

                    # Rude: {text}
                    # Polite:"""
                    
                    # result = corrector(
                    #     prompt,
                    #     max_length=100,
                    #     do_sample=True,
                    #     temperature=0.8,
                    #     top_p=0.9
                    # )
                    # inputs = tokenizer_t5(
                    #     prompt,
                    #     return_tensors="pt",
                    #     truncation=True
                    # )
                    # outputs = model_t5.generate(
                    #     **inputs,
                    #     max_length=100,
                    #     do_sample=True,
                    #     temperature=0.9,
                    #     top_p=0.95,
                    #     repetition_penalty=1.2
                    # )
                    # # suggestion = result[0]['generated_text']
                    # suggestion = tokenizer_t5.decode(outputs[0], skip_special_tokens=True)
                    # if any(word in suggestion.lower() for word in ["garbage", "waste", "useless", "bad"]):
                    #     suggestion = "Could you please review this and make the necessary improvements?"                        

                    # raw_result=corrector(few_shot_prompt, max_new_tokens=60, do_sample=True,temperature=0.7,max_length=None, top_p=0.9, return_full_text=False)[0]['generated_text']
                    
                    # if raw_result.startswith(few_shot_prompt):
                    #     suggestion = raw_result[len(few_shot_prompt):].strip()
                    # else:
                    #     suggestion = raw_result.strip()
                    
                # st.success(f"**Try sending this instead:**\n\n {suggestion}")
                # st.caption("✨ Generated using FLAN-T5 (text2text-generation) with prompt-based rewriting.")
            else:
                st.success("### ✅ Clear Communication!")
                st.write("The tone matches your intended emotion perfectly.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Core Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Your Intent", intended_emotion)
            with m2: st.metric("AI's Perception", final_emotion, delta="Adjusted" if is_sarcastic else "")
            with m3: st.metric("Sarcasm Detected", "Yes 😏" if is_sarcastic else "No")
            with m4: st.metric("Mismatch", "Yes ⚠️" if mismatch else "No ✅")

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
            
            # st.write("**BERT**")
            # st.progress(conf_bert)
            # st.caption(f"{conf_bert*100:.2f}%")
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