from fastapi import FastAPI, File, UploadFile
import uvicorn



from PIL import Image
import numpy as np

import tensorflow as tf


model = tf.keras.models.load_model('cnn.h5')
app = FastAPI()

@app.get("/")
async def index():
    return "Halo Bandung!"


@app.post("/")
async def predict(file: UploadFile=File(...)):
    image = Image.open(file.file)
    image = np.asarray(image.resize((150, 150)))
    image = np.expand_dims(image, 0)
    res = float(model.predict(image)[0][0])
    if res == 0:
        return "It's a Cat"
    else:
        return "It's a Dog" 

if __name__ == '__main__':
    uvicorn.run("myapi:app", host='127.0.0.1')