from google.cloud import storage
from fastapi import FastAPI
import torch
from diffusers import StableDiffusionXLPipeline
import os

app = FastAPI()

BUCKET_NAME = 'himeros-io.appspot.com'
MODEL_FILENAME = 'model.safetensors'
LOCAL_MODEL_PATH = f'/tmp/{MODEL_FILENAME}'

def download_model_from_gcs():
    if not os.path.exists(LOCAL_MODEL_PATH):
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILENAME)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print(f'Model downloaded to {LOCAL_MODEL_PATH}')
    else:
        print('Model already exists locally.')

@app.on_event("startup")
async def startup_event():
    download_model_from_gcs()
    # Load the model
    model = StableDiffusionXLPipeline.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.float16)
    model.to("cuda")
    app.state.model = model

@app.post("/generate")
async def generate(prompt: str):
    model = app.state.model
    image = model(prompt).images[0]
    # Save or process the image as needed
    return {"message": "Image generated successfully."}
