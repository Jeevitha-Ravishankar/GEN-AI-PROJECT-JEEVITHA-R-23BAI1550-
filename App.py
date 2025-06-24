import streamlit as st
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering, GPT2Tokenizer,
    GPT2LMHeadModel, pipeline
)
from gtts import gTTS
import torch
import os
import tempfile
import uuid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


@st.cache_resource
def load_captioning_models():
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return caption_processor, caption_model

@st.cache_resource
def load_vqa_models():
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return vqa_processor, vqa_model

@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model



@st.cache_resource
def load_emotion_pipeline():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

emotion_pipeline = load_emotion_pipeline()



if "stored_true_labels" not in st.session_state:
    st.session_state.stored_true_labels = []
if "stored_predicted_labels" not in st.session_state:
    st.session_state.stored_predicted_labels = []
if "stored_captions" not in st.session_state:
    st.session_state.stored_captions = []


def generate_caption(image, max_tokens):
    inputs = caption_processor(image, return_tensors="pt")
    with torch.no_grad():
        out = caption_model.generate(**inputs, max_new_tokens=max_tokens)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    return caption.strip().capitalize()

def answer_image_question(image, question):
    inputs = vqa_processor(image, question, return_tensors="pt")
    with torch.no_grad():
        output = vqa_model.generate(**inputs)
    answer = vqa_processor.decode(output[0], skip_special_tokens=True)
    return answer.strip().capitalize()

positive_words = {"laugh", "smile", "happy", "joy", "fun", "love", "beautiful", "sunny", "beach"}

def detect_mood(text):
    if any(word in text.lower() for word in positive_words):
        return "POSITIVE", 0.99

    results = emotion_pipeline(text)
    label = results[0]['label'].upper()
    score = results[0]['score']

    positive_emotions = {"JOY", "LOVE", "SURPRISE"}
    negative_emotions = {"ANGER", "FEAR", "SADNESS"}

    if label in positive_emotions:
        return "POSITIVE", score
    elif label in negative_emotions:
        return "NEGATIVE", score
    else:
        return "NEUTRAL", score

def generate_hashtags(caption):
    keywords = caption.lower().replace(".", "").split()
    stopwords = {"in", "at", "the", "a", "and", "of", "on", "with", "together", "is"}
    tags = [f"#{word}" for word in keywords if word not in stopwords]
    return " ".join(tags[:5])

def print_evaluation_to_console(true_labels, predicted_labels):
    if len(true_labels) == 0:
        print("No stored labels to evaluate yet.")
        return

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print("\n----- Mood Detection Evaluation Results (Console Only) -----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    labels = sorted(list(set(true_labels + predicted_labels)))
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)

    import tempfile
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp_file.name)
    plt.close(fig)

    print(f"Confusion matrix plot saved to: {tmp_file.name}")



caption_processor, caption_model = load_captioning_models()
vqa_processor, vqa_model = load_vqa_models()
gpt2_tokenizer, gpt2_model = load_gpt2()

# --- UI ---

st.set_page_config(page_title="GEN AI Caption Bot", page_icon="🖼️")
st.title("🖼️ImagiQ: Smart Captioning, Hashtags, Mood Detection & Visual Q&A")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
quality = st.slider("🎛️ Caption Quality (Higher = more descriptive)", 10, 60, 30)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔎 Generating caption..."):
        caption = generate_caption(image, quality)

    st.success("📝 Caption:")
    st.markdown(f"### {caption}")

    st.session_state.stored_captions.append(caption)

    # --- TTS ---
    if st.button("🔊 Play Caption"):
        tts = gTTS(caption, lang='en')
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
        tts.save(temp_path)
        with open(temp_path, "rb") as f:
            st.audio(f.read(), format="audio/mp3")
        os.remove(temp_path)

    # --- Download Caption ---
    st.download_button("⬇️ Download Caption", caption, file_name="caption.txt")

    # --- Hashtag Generator ---
    if st.button("🏷️ Generate Hashtags"):
        hashtags = generate_hashtags(caption)
        st.markdown("**🏷️ Suggested Hashtags:**")
        st.code(hashtags)

    # --- Mood Detection ---
    st.markdown("### 🧠 Mood Detection")
    true_label_input = st.text_input("Enter TRUE mood label for this caption (e.g. POSITIVE, NEGATIVE)")

    if st.button("🧠 Detect Mood from Caption"):
        label, score = detect_mood(caption)
        st.markdown(f"**🧠 Detected Mood:** {label} (Confidence: {score:.2f})")

        if true_label_input:
            true_label = true_label_input.strip().upper()
            pred_label = label.upper()
            st.session_state.stored_true_labels.append(true_label)
            st.session_state.stored_predicted_labels.append(pred_label)
            st.success(f"Stored true label '{true_label}' and predicted label '{pred_label}' for evaluation.")
            print_evaluation_to_console(st.session_state.stored_true_labels, st.session_state.stored_predicted_labels)
        else:
            st.warning("Please enter a true label before clicking the button.")

    # --- Q&A ---
    st.markdown("---")
    st.subheader("💬 Ask a question about the image")
    question = st.text_input("e.g. What color is the umbrella?")
    if question:
        with st.spinner("🤔 Thinking..."):
            answer = answer_image_question(image, question)
        st.success(f"🤖 Answer: {answer}")
else:
    st.info("📷 Please upload an image to begin.")
