
import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image

# --- โหลดโมเดล SVM ---
model = joblib.load("svm_image_classifier_model.pkl")

# --- สร้าง UI ---
st.title("Fruit Classifier")
st.write("Please upload file for predict 🤩(Apple and Orange)")

# dictionary แปลง class index เป็นชื่อ
class_dict = {0: "🍎APPLE🍎", 1: "🍊ORANGE🍊"}

# --- อัปโหลดรูป ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # อ่านและแปลงเป็น RGB
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # ปุ่มทำนาย
    if st.button("Predict"):
        # --- แปลงรูปเป็น array ตามตอน train ---
        image_array = np.array(image)
        # ถ้า train ใช้ OpenCV default BGR ให้ convert
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        # Resize เป็น (100, 100) ตามตอน train
        image_resized = cv2.resize(image_array, (100, 100))
        # Flatten เป็น feature vector
        image_flatten = image_resized.flatten().reshape(1, -1)

        # --- ทำนาย ---
        prediction = model.predict(image_flatten)[0]
        prediction_name = class_dict[prediction]

        st.write(f"It is a : {prediction_name}")
