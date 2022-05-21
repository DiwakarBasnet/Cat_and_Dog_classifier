from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("./Models/binary_cat_dog_classifier")
CLASS_NAMES = ["Cat", "Dog"]

@app.get("/ping")
async def ping():
    return "Hello, I am awake"


def read_file_as_image(data) -> np.ndarray:                 # function takes bytes as input and sends output in ndarray
    image = np.array(Image.open(BytesIO(data)))             # BytesIO sends bytes to PIL module
                                                            # Image.open reads those bytes as PILO image
                                                            # np.array converts PILO image to numpy array
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())           # reads file sent as bytes and sends to function called
    img_batch = np.expand_dims(image, 0)                    # adds another dimension to array since MODEL.predict takes a 
                                                            # batch instead of single image, 0 is axis of array
    predictions = MODEL.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)