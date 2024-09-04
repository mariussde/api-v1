import io
import pickle
import numpy as np
from PIL import Image
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

with open('mnist_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post('/predict-image')
async def predict(file: UploadFile = File(...)):
    pil_image = PIL.Image.open(io.BytesIO(await file.read())).convert('L')
    pil_image = PIL.ImageOps.invert(pil_image)
    pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.array(pil_image).reshape(1, -1)
    prediction = model.predict(image_array)
    return {'prediction': int(prediction[0])}
