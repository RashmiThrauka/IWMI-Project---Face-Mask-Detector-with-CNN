import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import cv2

model = keras.models.load_model('best_model.h5')
classes = ['with_mask', 'without_mask']

st.set_page_config(page_title="Face Mask Detector", layout="centered")
st.title("Face Mask Detector")
st.write("Upload a face image to detect whether the person is wearing a mask.")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        face_img = cv2.resize(img_array, (128, 128)) / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        pred = model.predict(face_img, verbose=0)[0]
        label = classes[np.argmax(pred)]
        confidence = np.max(pred) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded Image")
        with col2:
            st.metric("Prediction", label)
            st.metric("Confidence", f"{confidence:.1f}%")

    else:
        img_draw = img_array.copy()
        for (x, y, w, h) in faces:
            face = img_array[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (128, 128)) / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)
            pred = model.predict(face_resized, verbose=0)[0]
            label = classes[np.argmax(pred)]
            confidence = np.max(pred) * 100

            color = (0, 255, 0) if label == 'with_mask' else (255, 0, 0)
            cv2.rectangle(img_draw, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_draw, f'{label} {confidence:.1f}%',
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_draw, caption="Detection Result")
        with col2:
            st.metric("Prediction", label)
            st.metric("Confidence", f"{confidence:.1f}%")

    fig, ax = plt.subplots()
    ax.bar(classes, pred * 100, color=['green', 'red'])
    ax.set_ylabel("Confidence %")
    ax.set_title("Top Predictions")
    st.pyplot(fig)

with st.sidebar:
    st.header("Model Info")
    st.write("**Architecture:** Custom CNN")
    st.write("**Layers:** 3x Conv2D + BatchNorm + MaxPool, Dropout, 2x Dense")
    st.write("**Test Accuracy:** 96%")
    st.write("**Dataset:** with_mask / without_mask")