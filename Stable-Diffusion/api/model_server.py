
from fastapi import FastAPI, File, UploadFile

import numpy as np
import torch

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

device = 'cuda'

image = Image.open("results/penguin_output/penguins.png")
mask_image = Image.open("results/penguin_output/summed_mask_good.png")

prompt = "Otters"

#Stable Diffusion Pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    )
pipe = pipe.to(device) 

image = pipe(prompt=prompt, image=image, mask_image=mask_image)

image.save("results/penguin_output/penguins_diffused.png")