from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from diffusers import AutoPipelineForInpainting, UNet2DConditionModel
import torch
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
import diffusers

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
API_KEY_NAME = "X-API-Key"
API_KEY = "your-secret-api-key"  # Replace with your actual API key

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

class InpaintRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    guidance_scale: float = 7.5
    steps: int = 20
    strength: float = 1.0
    scheduler: str = "EulerDiscreteScheduler"

@app.on_event("startup")
async def startup_event():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoPipelineForInpainting.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        app.state.model = model
        app.state.device = device
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")

@app.post("/inpaint", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")  # Limit to 5 requests per minute
async def inpaint(
    request: InpaintRequest,
    image: UploadFile = File(...),
    mask: UploadFile = File(...)
):
    model = app.state.model
    device = app.state.device

    try:
        init_image = Image.open(BytesIO(await image.read())).convert("RGB").resize((1024, 1024))
        mask_image = Image.open(BytesIO(await mask.read())).convert("RGB").resize((1024, 1024))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image or mask file")

    try:
        negative_prompt = request.negative_prompt if request.negative_prompt else None
        scheduler_class_name = request.scheduler.split("-")[0]

        add_kwargs = {}
        if len(request.scheduler.split("-")) > 1:
            add_kwargs["use_karras"] = True
        if len(request.scheduler.split("-")) > 2:
            add_kwargs["algorithm_type"] = "sde-dpmsolver++"

        scheduler = getattr(diffusers, scheduler_class_name)
        model.scheduler = scheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", **add_kwargs
        )

        output = model(
            prompt=request.prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.steps,
            strength=request.strength
        )

        # Save or process the image as needed
        return {"message": "Image generated successfully."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Image generation failed: {e}")

# Add exception handler for rate limiting
@app.exception_handler(429)
async def rate_limit_exceeded_handler(request, exc):
    return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS, content={"detail": "Rate limit exceeded"})
