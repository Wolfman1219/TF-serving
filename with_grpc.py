import numpy as np
import streamlit as st
from PIL import Image
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import json
from numpy import ndarray
from typing import List, Optional, Tuple, Union
import cv2
import tempfile
import time

####LETTERBOX###########
def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (0, 0, 0)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)

##########BLOB###########
def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im
#######################################################################
##############NMS#####################################################
def nms(boxes, scores, iou_threshold):
    # Convert to xyxy
    boxes = xywh2xyxy(boxes)
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest 
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over threshold
        keep_indices = np.where(ious < iou_threshold)[0] + 1

        sorted_indices = sorted_indices[keep_indices]

    return keep_boxes
# def nms(boxes, scores, iou_threshold):
#     # Sort by score
#     sorted_indices = np.argsort(scores)[::-1]

#     keep_boxes = []
#     while sorted_indices.size > 0:
#         # Pick the last box
#         box_id = sorted_indices[0]
#         keep_boxes.append(box_id)

#         # Compute IoU of the picked box with the rest
#         ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

#         # Remove boxes with IoU over the threshold
#         keep_indices = np.where(ious < iou_threshold)[0]

#         # print(keep_indices.shape, sorted_indices.shape)
#         sorted_indices = sorted_indices[keep_indices + 1]
    
#     return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# Function to send an image to the TensorFlow Serving server for prediction using gRPC
def get_input(model_name):
     # Get the serving_input key
    loaded_model = tf.saved_model.load(model_name)
    input_name = list(
        loaded_model.signatures["serving_default"].structured_input_signature[1].keys()
    )[0]
    return input_name

def predict_image_grpc(image, channel, input_name):
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'yolov8n'  # Replace with your model name
    request.model_spec.signature_name = 'serving_default'
    request.inputs[input_name].CopyFrom(tf.make_tensor_proto(image, dtype=tf.float32))  # Adjust the input tensor name if needed
    response = stub.Predict(request)
    return response

# Streamlit app
st.title("Object detection with yolov8n")

# Upload an image
input_name = get_input("/home/hasan/Public/TF-serving/model/yolov8n")

channel = grpc.insecure_channel("localhost:3939", options=(('grpc.enable_http_proxy', 0),))  # Adjust the server address and port if needed
def detect_image(img):
    # Display the uploaded image
    # Preprocess the image
    # img = Image.open(uploaded_image)  # Convert to grayscale
    input_shape = (640, 640)  # Replace with the expected input shape of your YOLO model
    bgr, ratio, dwdh = letterbox(np.array(img), input_shape)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    tensor = np.ascontiguousarray(tensor)

    # Create a gRPC channel to TensorFlow Serving

    # Make a prediction using gRPC
    response = predict_image_grpc(tensor, channel, input_name)
    predictions = response.outputs['output0'].float_val  # Adjust the output tensor name if needed
    predictions = np.array(predictions).reshape((84, 8400))
    predictions = predictions.T 


########################
    conf_thresold = 0.5
    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_thresold, :]
    scores = scores[scores > conf_thresold]  
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = predictions[:, :4]

    #rescale box
    input_shape = np.array([640, 640, 640, 640])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([640, 640, 640, 640])
    boxes = boxes.astype(np.int32)
    iou_thres = 0.1
    indices = nms(boxes, scores, iou_thres)
    image_draw = rgb.copy()
    for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
        bbox = bbox.round().astype(np.int32).tolist()
        cls_id = int(label)
        # cls = CLASSES[cls_id]
        color = (0,255,0)
        cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
        cv2.putText(image_draw,
                    f'car:{int(score*100)}', (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, [225, 255, 255],
                    thickness=1)
    return image_draw
    # cv2.imwrite("image_output.jpg",image_draw)

def main():
    DEFAULT_VIDEO_PATH = "data/sample_videos/sample.mp4"
# Create a video file uploader
    st.header("Upload a video for inference")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    # Create a radio button for selecting between default video and uploaded video
    video_selection = st.radio(
        "Select video for inference:",
        ("Use default video", "Use uploaded video")
    )

    # If the user chooses to use the default video
    if video_selection == "Use default video":
        video_path = DEFAULT_VIDEO_PATH

    # If the user chooses to use the uploaded video
    elif video_selection == "Use uploaded video" and uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    # If there's a video to process, do the inference
    if video_path is not None:
        # Load the video with cv2
        cap = cv2.VideoCapture(video_path)
        for_fps = st.empty()
        outputing = st.empty()
        
        fps = 0
        prev_time = 0
        curr_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run the inference
            output = detect_image(frame)

            # Convert the output to an image that can be displayed
            output_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

            # Display the image
            outputing.image(output_image)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            for_fps.write(f"FPS: {fps}")
            # print(fps)
        cap.release()
    else:
        st.write("Please upload a video file or choose to use the default video.")

if __name__ == "__main__":
    main()
    
