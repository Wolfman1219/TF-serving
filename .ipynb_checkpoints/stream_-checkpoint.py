import json
import requests
import numpy as np
import streamlit as st
from PIL import Image

# Function to send an image to the TensorFlow Serving server for prediction
def predict_image(image):
    # Prepare the data for prediction
    data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})
    headers = {"content-type": "application/json"}

    # Send POST request to the server
    response = requests.post("http://localhost:3939/v1/models/mnist_model:predict", data=data, headers=headers)

    # Parse and return the response
    predictions = json.loads(response.text)
    return predictions

# Streamlit app
st.title("MNIST Image Classifier")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = Image.open(uploaded_image).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to MNIST input size
    img = np.array(img) / 255.0  # Normalize pixel values

    # Make a prediction
    predictions = predict_image(img)

    # Display the prediction result
    st.subheader("Prediction Result:")
    for i, prediction in enumerate(predictions["predictions"]):
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        st.write(f"Image - Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
