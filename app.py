import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import tempfile

# Load model
model = tf.keras.models.load_model("violence_model.h5")

def preprocess(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

st.set_page_config(page_title="Video Surveillance System")

st.title("🎥 Video Surveillance System")

option = st.sidebar.selectbox(
    "Choose Input Type",
    ["Upload Video", "Upload Image", "Live Camera"]
)

# ---------------- VIDEO ----------------
if option == "Upload Video":
    st.header("📂 Upload Video")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        predictions = []
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            input_frame = preprocess(frame)
            pred = model.predict(input_frame, verbose=0)[0][0]
            predictions.append(pred)

            stframe.image(frame, channels="BGR")

        cap.release()

        avg_pred = np.mean(predictions)

        if avg_pred > 0.5:
            st.error("✅ NORMAL")
        else:
            st.success("⚠️ ABNORMAL (Violence Detected)")

# ---------------- IMAGE ----------------
elif option == "Upload Image":
    st.header("🖼 Upload Image")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR")

        input_frame = preprocess(image)
        pred = model.predict(input_frame, verbose=0)[0][0]

        if pred > 0.5:
            st.error("✅ NORMAL")
        else:
            st.success("⚠️ ABNORMAL (Violence Detected)")

# ---------------- CAMERA ----------------
elif option == "Live Camera":
    st.header("📷 Live Camera")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    predictions = []

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        input_frame = preprocess(frame)
        pred = model.predict(input_frame, verbose=0)[0][0]

        predictions.append(pred)
        if len(predictions) > 10:
            predictions.pop(0)

        avg_pred = np.mean(predictions)

        if avg_pred > 0.7:
            label = "NORMAL ✅"
            color = (0, 0, 255)
        else:
            label = "ABNORMAL ⚠️"
            color = (0, 255, 0)

        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()