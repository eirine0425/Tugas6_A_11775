import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Load model
model_path = '/content/drive/MyDrive/Colab Notebooks/Tugas6_A_11775/model_mobilenet.keras'
try:
    model = load_model(model_path)
    class_names = ['Matang', 'Mentah']
except Exception as e:
    st.error(f"Model gagal dimuat. Pastikan path file model benar. Error: {e}")
    st.stop()

# Function to classify images
def classify_image(image):
    try:
        # Preprocess the image
        input_image = image.resize((180, 180))  # Resize directly using PIL
        input_image_array = np.array(input_image) / 255.0  # Normalize
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)  # Add batch dimension
        
        # Predict
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])  # Apply softmax
        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

# Custom progress bar
def custom_progress_bar(confidence, color1, color2):
    percentage1 = confidence[0] * 100
    percentage2 = confidence[1] * 100
    progress_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;">
        <div style="width: {percentage1:.2f}%; background: {color1}; color: white; text-align: center; height: 24px; float: left;">
            {percentage1:.2f}%
        </div>
        <div style="width: {percentage2:.2f}%; background: {color2}; color: white; text-align: center; height: 24px; float: left;">
            {percentage2:.2f}%
        </div>
    </div>
    """
    st.sidebar.markdown(progress_html, unsafe_allow_html=True)

# Streamlit App UI
st.title("Prediksi Kematangan Buah Naga - 1775")

uploaded_files = st.file_uploader("Unggah Gambar (JPG/PNG/JPEG diperbolehkan)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            try:
                # Open the uploaded file as an image
                image = Image.open(uploaded_file)
                label, confidence = classify_image(image)

                if label != "Error":
                    primary_color = "#007BFF"
                    secondary_color = "#FF4136"
                    label_color = primary_color if label == "Matang" else secondary_color
                    st.sidebar.write(f" * Nama File :* {uploaded_file.name}")
                    st.sidebar.markdown(f"<h4 style='color: {label_color};'>Prediksi: {label}</h4>", unsafe_allow_html=True)
                    st.sidebar.write(" * Confidence :* ")
                    for i, class_name in enumerate(class_names):
                        st.sidebar.write(f"- {class_name}: {confidence[i] * 100:.2f}%")
                    custom_progress_bar(confidence, primary_color, secondary_color)
                    st.sidebar.write(" --- ")
                else:
                    st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
            except Exception as e:
                st.sidebar.error(f"Kesalahan saat membuka gambar {uploaded_file.name}: {e}")
    else:
        st.sidebar.error("Silakan unggah setidaknya satu gambar untuk diprediksi.")

if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
        except Exception as e:
            st.error(f"Kesalahan saat menampilkan gambar {uploaded_file.name}: {e}")
