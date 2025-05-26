from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
from PIL import Image
import io

app = FastAPI()

classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = classifier(image)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar la imagen: {str(e)}")
