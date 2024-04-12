from fastapi import FastAPI, UploadFile, File

import os
import sys
import logging

pth = '/home/ommos92/adv-computer-vision/SVLM/Segment-Everything-Everywhere-All-At-Once/'
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(0)
import cv2

import torch
from torchvision import transforms

from utils.arguments import load_opt_command
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from modeling.BaseModel import BaseModel
from modeling import build_model
from detectron2.utils.colormap import random_color
from utils.visualizer import Visualizer
from utils.distributed import init_distributed

app = FastAPI()

# Load the model on startup
@app.on_event("startup")
def load_model():
    global model
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir
    opt = init_distributed(opt)

    pretrained_pth = os.path.join(opt['RESUME_FROM'])
    if 'novg' not in pretrained_pth:
        assert False, "Using the ckpt without visual genome training data will be much better."
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["background"], is_eval=False)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t = []
    t.append(transforms.Resize(224, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    with torch.no_grad():
        image_ori = Image.open(file.file).convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width, 'image_id': 0}]
        outputs = model.model.evaluate_captioning(batch_inputs)
        text = outputs[-1]['captioning_text']

        image_ori = image_ori[:,:,::-1].copy()
        cv2.rectangle(image_ori, (0, 0), (width, 60), (0,0,0), -1)
        font                   = cv2.FONT_HERSHEY_DUPLEX
        fontScale              = 1.2
        thickness              = 2
        lineType               = 2
        bottomLeftCornerOfText = (10, 40)
        fontColor              = [255,255,255]
        cv2.putText(image_ori, text,
            bottomLeftCornerOfText,
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        output_root = './output'
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        cv2.imwrite(os.path.join(output_root, 'captioning.png'), image_ori)

    return {"caption": text}

@app.on_event("shutdown")
def shutdown_event():
    del model

