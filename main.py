import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle

# Load the history object
with open('history.pkl', 'rb') as file:
    loaded_history = pickle.load(file)

# Loading the Model
model = load_model('plant_disease.h5')

# Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Page title and description
st.markdown(
    """
    <style>
        .container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 50px;
        }
        .left-content {.streamlit run main.py
            flex: 1;
            padding-right: 40px;
        }
        .right-content {
            flex: 1;
            padding-left: 40px;
            display: flex;
            justify-content: flex-end;
            align-items: center;
        }
        .logo {
            width: 300px;
            height: auto;
            margin-left: 20px;
            animation: spin 3s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>

    <div class="container">
        <div class="left-content">
            <h1>Detect Plant Diseases with AI</h1>
            <p>
                Welcome to our Plant Disease Detection website powered by artificial intelligence. 
                Upload an image of a plant leaf, and our advanced AI model will analyze it to identify 
                any signs of common diseases. This technology enables early detection, prevention, and 
                effective treatment of plant diseases, contributing to healthier crops and increased 
                agricultural productivity.
            </p>
        </div>
        <div class="right-content">
            <img class="logo" src="https://user-images.githubusercontent.com/30645315/68544440-37ffdd80-03e9-11ea-8acd-3f3f9b6fc8b3.png" alt="Logo">
        </div>
    </div>

    <hr style='border-top: 2px solid #177E89; margin-top: 30px; margin-bottom: 30px;'>

    <h2>Upload Image</h2>
    <p>Please upload an image of a plant leaf in JPG format.</p>
    <p>Our AI model will analyze the image and accurately identify any signs of disease.</p>
    """,
    unsafe_allow_html=True
)

# Uploading the image
plant_image = st.file_uploader("Choose an image...", type="jpg")

# Prediction button
if st.button('Predict'):
    if plant_image is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(opencv_image, channels="BGR", caption="Uploaded Image")

        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # Convert image to 4 Dimensions
        opencv_image.shape = (1, 256, 256, 3)

        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.markdown(f"<h3>Result: {result}</h3>", unsafe_allow_html=True)

# Model accuracy section
st.markdown("<h2>Model Accuracy</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <p>Check the accuracy of our AI model in identifying plant diseases.</p>
    <p>The plot below shows the training and validation accuracy over epochs.</p>
    """,
    unsafe_allow_html=True
)

# Create the plot
plt.figure(figsize=(12, 5))
plt.plot(loaded_history['accuracy'], color='#177E89', label='Training Accuracy')
plt.plot(loaded_history['val_accuracy'], color='#F78536', label='Validation Accuracy')
plt.title("Model Accuracy", fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.xlabel('Epochs', fontweight='bold')
plt.legend()

# Display the plot
st.pyplot(plt)