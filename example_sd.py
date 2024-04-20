import torch
from diffusers import StableDiffusionInpaintPipeline

device = 'cuda:3'

#Stable Diffusion Pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    )
pipe = pipe.to(device)  # Send the diffusion model to the GPU


image = Image.fromarray(image)
mask = Image.fromarray(mask)

image = image.resize((512, 512))
mask = mask.resize((512, 512))

output = pipe(prompt=prompt, image=image, mask=mask).images[0]