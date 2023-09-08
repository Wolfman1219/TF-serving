# TF-serving
A web interface for real-time yolo inference using streamlit. It supports CPU and GPU inference, support videos and uploading your own custom videos.

<img src="view.gif" alt="demo of the dashboard" width="800"/>


## Features
- Supports uploading model files (<200MB) and downloading models from URL (any size)
- Supports videos.

## How to run
After cloning the repo:
1. Install requirements
   - `pip install -r requirements.txt`
2. Add sample images to `data/sample_images`
3. Add sample video to `data/sample_videos` and call it `sample.mp4` or change name in the code.
4. Add the model file to `model/` and change code to its path.
```bash
git clone https://github.com/Wolfman1219/TF-serving.git
cd TF-serving
sudo docker run -p 3939:8500 --name=tf_model_serving   --mount type=bind,source=./model/yolov8n,target=/models/yolov8n/1   -e MODEL_NAME=yolov8n -t tensorflow/serving
streamlit run app.py
```
