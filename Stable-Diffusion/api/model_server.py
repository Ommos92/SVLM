from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import numpy as np
from typing import List

app = FastAPI()

device = 'cuda'

@app.post("/process_image")
async def process_image(image_path: str, mask_image: List[List[float]], prompt: str, result_path : str):
    # Load the pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)

    # Read the image from the provided path
    image = Image.open(image_path).convert("RGB")

    # Convert the list back to a numpy array
    mask_image = np.array(mask_image)

    # Process the image
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        strength=0.95,
    ).images[0]

    # Save the result
    result_path = "results/penguin_output/penguins_diffused.png"
    result.save(result_path)

    # Return the result as a file response
    return FileResponse(result_path)