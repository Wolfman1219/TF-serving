from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
import tensorflow as tf
from PIL import Image

MODEL = tf.keras.models.load_model("model/mnist_model.h5")
CLASS_NAMEs = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
app = FastAPI()
def read_files_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
    
@app.get("/ping")
async def ping():
    return "Hello world"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_files_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    prediction = MODEL.predict(image)
    prediction_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {'class': prediction_class, 'confidence':float(confidence)} 
    

if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=5004)
    