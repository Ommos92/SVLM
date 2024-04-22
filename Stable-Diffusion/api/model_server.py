
from fastapi import FastAPI, File, UploadFile

import numpy as np
import torch

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

device = 'cuda'

image = Image.open("SEEM/inference/images/penguin.jpeg").convert("RGB")
mask_image = Image.open("results/penguin_output/summed_mask_good.png").convert("RGB")

prompt = "Personify the penguins in the image."

#Stable Diffusion Pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    )
pipe = pipe.to(device) 

image = pipe(
    prompt=prompt, 
    image=image, 
    mask_image=mask_image,
    strength=0.95,
    ).images[0]

image.save("results/penguin_output/penguins_diffused.png")