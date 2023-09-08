import json
import requests
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import json

def draw_boxes(image = np.random.rand(3, 640, 640), detection_output = np.random.rand(1, 5, 8400), confidence_threshold=0.1):
    h, w, _ = image.shape
    boxes = []

    for detection in detection_output[0]:
        class_probabilities = detection[5:]
        class_id = np.argmax(class_probabilities)
        confidence = class_probabilities[class_id]
        print("\nconfidence\t",confidence, "\nclass_probabilities\t",class_probabilities)
        if confidence > confidence_threshold:
            center_x, center_y, width, height = detection[:4]
            
            # Convert coordinates from relative to absolute
            center_x = int(center_x * w)
            center_y = int(center_y * h)
            width = int(width * w)
            height = int(height * h)
            
            # Calculate top-left and bottom-right coordinates of the bounding box
            top_left_x = center_x - width // 2
            top_left_y = center_y - height // 2
            bottom_right_x = center_x + width // 2
            bottom_right_y = center_y + height // 2
            
            # Append the box information to the list
            boxes.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y, class_id, confidence))
    
    # Draw bounding boxes on the image
    for box in boxes:
        top_left_x, top_left_y, bottom_right_x, bottom_right_y, class_id, confidence = box
        # top_left, bottom_right, class_id, confidence = box[:4], box[4:6]
        color = (0, 255, 0)  # Green color
        label = f'Class name: {confidence:.2f}'
        
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
        
        # Draw the class label
        cv2.putText(image, label, (top_left_x, top_left_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image


# Function to send an image to the TensorFlow Serving server for prediction
def predict_image(image):
    # Prepare the data for prediction
    data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})
    headers = {"content-type": "application/json"}

    # Send POST request to the server
    response = requests.post("http://localhost:3939/v1/models/yolov8n:predict", data=data, headers=headers)
    print(response)
    # Parse and return the response
    predictions = json.loads(response.text)
    return predictions

# Streamlit app
st.title("YOLO prediction")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    # Convert to grayscale
    img = Image.open(uploaded_image).convert("RGB")  # Convert to RGB
    input_shape = (640, 640)  # Replace with the expected input shape of your YOLO model
    img = img.resize(input_shape)
    im = np.stack([img])
    im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
    im = im.astype(np.float32) / 255.0
    im = np.ascontiguousarray(im)  # contiguous
    # im = ov.Tensor(array=im, shared_memory=True)
    # Pad the image to match the expected input shape
    # Make a prediction
    predictions = predict_image(im)
    to_arr = np.array(predictions['predictions'])
    print(to_arr.shape)
    with open("predictions.json", "w") as outfile:
        json.dump(predictions, outfile)
    image = draw_boxes(np.array(img), np.array(predictions['predictions']))
    st.image(image, caption="Result Image", use_column_width=True)
    cv2.imwrite("result_image.jpg", image)  
    # Display the prediction result
    # st.subheader("Prediction Result:")
    # for i, prediction in enumerate(predictions["predictions"]):
    #     predicted_class = np.argmax(prediction)
    #     confidence = prediction[predicted_class]
    #     st.write(f"Image - Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
