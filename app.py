
import os

import gradio as gr
import numpy as np
import torch

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


#from SEEM.modeling.BaseModel import BaseModel
#from SEEM.modeling import build_model
#from SEEM.utils.distributed import init_distributed
#from SEEM.utils.arguments import load_opt_from_config_files
#from SEEM.utils.constants import COCO_PANOPTIC_CLASSES

#from SEEM.demo.seem.tasks import *

# device = "cuda"
# SEEM_checkpoint = "SEEM/weights/seem_focall_v0.pt"

# opt, cmdline_args = load_opt_command(args)
# if cmdline_args.user_dir:
#     absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
#     opt['base_path'] = absolute_user_dir

# opt = init_distributed(opt)

# # SEEM Model Initialization
# pretrained_pth = os.path.join(opt['RESUME_FROM'])
# output_root = 'results/penguin_output'
# image_pth = 'SEEM/inference/images/penguin.jpeg'

# model = BaseModel(opt, build_model(opt)).from_pretrained(load_dir='SEEM/weights/seem_focall_v0.pt').eval().cuda()
# with torch.no_grad():
#     model.model.sem_seg_head

device = 'cuda:3'

#Stable Diffusion Pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    )
pipe = pipe.to(device)  # Send the diffusion model to the GPU


# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        image = gr.Image(type="Input image", label="Image")
        task = gr.Image(label="Mask")
        output = gr.Image(label="Output")

    with gr.Block():
        prompt_text = gr.Textbox(lines=1, label="Prompt", placeholder="Enter a prompt to describe the image")

    with gr.Row():
        submit = gr.Button(label="Submit")
        



def generate_mask(input_img, caption, evt: gr.SelectData):
    # May want to use a caption prompt to generate the mask
    prompt = prompt_text.value

    #Call the SEEM Model
    #mask_img = segment_everything_everywhere(model, input_img, prompt)
    mask_img = np.zeros_like(input_img)
    
    


def inpaint(image, mask, prompt):
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    image = image.resize((512, 512))
    mask = mask.resize((512, 512))

    output = pipe(prompt=prompt, image=image, mask=mask).images[0]
    return output

input_img.select(generate_mask, [input_img], [mask_img])