# app.py
import os
import uuid
import streamlit as st
import pandas as pd
from PIL import Image

from models.imagecaption import ImageCaptioner
from models.text_classification import ToxicityClassifier
from data_base.database import CSVDatabase

TOX_MODEL_PATH = r"D:\Cellula NLP Training\week 2\Task1\models\Distil-BERT_model"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@st.cache_resource(show_spinner=False)
def init_captioner():
    return ImageCaptioner()

@st.cache_resource(show_spinner=False)
def init_toxicity():
    return ToxicityClassifier(model_path=TOX_MODEL_PATH)

captioner = init_captioner()
tox_classifier = init_toxicity()
db = CSVDatabase(csv_path="combined_data.csv")

st.title(" Toxic Content Classifier (Text & Image) ")

tab1, tab2 = st.tabs(["Classify Input", "View Database"])

with tab1:
    option = st.radio("Choose input type", ["Text", "Image"])

    if option == "Text":
        user_text = st.text_area("Enter text here:")
        if st.button("Classify Text"):
            if not user_text or not user_text.strip():
                st.warning("Please enter text before classifying.")
            else:
                result = tox_classifier.predict(user_text)
                st.success(f"Class: **{result['predicted_class']}**  |  Confidence: **{result['confidence_score']:.4f}**")
                db.insert_record("text", user_text, result['predicted_class'], result['confidence_score'])

    else:  
        uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])
        if st.button("Classify Image"):
            if uploaded_image is None:
                st.warning("Please upload an image first.")
            else:
              
                unique_name = f"{uuid.uuid4().hex}_{uploaded_image.name}"
                save_path = os.path.join(UPLOAD_DIR, unique_name)
                pil_img = Image.open(uploaded_image).convert("RGB")
                pil_img.save(save_path)

                st.image(pil_img, caption="Uploaded image", use_column_width=True)

                # Caption generation
                caption = captioner.generate(pil_img)
                st.info(f"Generated caption: {caption}")

                # Classification
                result = tox_classifier.predict(caption)
                st.success(f"Class: **{result['predicted_class']}**  |  Confidence: **{result['confidence_score']:.4f}**")

                db.insert_record("image", caption, result['predicted_class'], result['confidence_score'])

with tab2:
    st.subheader("Stored records (CSV)")
    rows = db.fetch_all()
    if not rows:
        st.info("No records yet.")
    else:
        df = pd.DataFrame(rows, columns=["Original Input", "Caption", "Predicted Class", "Confidence"])
        try:
            df["Confidence"] = df["Confidence"].astype(float)
        except Exception:
            pass
        st.dataframe(df)
        if st.button("Download CSV"):
            with open(db.csv_path, "rb") as f:
                st.download_button("Download CSV file", f, file_name=os.path.basename(db.csv_path))
