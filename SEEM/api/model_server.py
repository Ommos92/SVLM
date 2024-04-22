from typing import List

from fastapi import FastAPI, UploadFile, File

import os
import sys
import logging
import argparse

pth = '/home/ommos92/adv-computer-vision/SVLM/Segment-Everything-Everywhere-All-At-Once/'
sys.path.insert(0, pth)

from fastapi import FastAPI
from uvicorn import run
import pydantic 


import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from modeling.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

import cv2
import os
import glob
import subprocess
from PIL import Image
import random


from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.visualizer import Visualizer
from utils.distributed import init_distributed


args = None
# SEEM Model load args 
opt, cmdline_args = load_opt_command(args)
if cmdline_args.user_dir:
    absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
    opt['base_path'] = absolute_user_dir
# Check if 'device' key is in the opt dictionary
if 'device' not in opt:
    # If not, add it
    opt['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre trained Path
pretrained_pth =  'SEEM/weights/seem_focall_v0.pt'
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

#Declare the transform
t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)

app = FastAPI()
#run(app, host="0.0.0.0", port = 8000, reload=True)

class Item(pydantic.BaseModel):
    image_pth : str = pydantic.Field(..., example='SEEM/inference/images/street.jpg')
    thing_classes: List[str] = pydantic.Field(..., example=['car','person','traffic light', 'truck', 'motorcycle'])
    stuff_classes: List[str] = pydantic.Field(..., example=['building','sky','street','tree','rock','sidewalk'])
    save_image: bool = pydantic.Field(..., example=True)            #Save the image
    output_root: str = pydantic.Field(..., example="path/to/output")

def load_image(image_pth):
    image = Image.open(image_pth)
    image = transform(image)
    image = transforms.ToTensor()(image).unsqueeze(0).cuda()
    return image


@app.post("/predict_panoseg")
async def predict(item: Item):

    thing_classes = item.thing_classes
    stuff_classes = item.stuff_classes
    thing_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(thing_classes))]
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(stuff_classes))]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
    stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes + ["background"], is_eval=False)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)

    with torch.no_grad():
        image_ori = Image.open(item.image_pth).convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs)


        #Save the output
        if item.save_image == True:
            visual = Visualizer(image_ori, metadata=metadata)
            pano_seg = outputs[-1]['panoptic_seg'][0]
            pano_seg_info = outputs[-1]['panoptic_seg'][1]

            for i in range(len(pano_seg_info)):
                if pano_seg_info[i]['category_id'] in metadata.thing_dataset_id_to_contiguous_id.keys():
                    pano_seg_info[i]['category_id'] = metadata.thing_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                else:
                    pano_seg_info[i]['isthing'] = False
                    pano_seg_info[i]['category_id'] = metadata.stuff_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]

            demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image

            if not os.path.exists(item.output_root):
                os.makedirs(item.output_root)
            demo.save(os.path.join(item.output_root, 'pano.png'))
    
    return {"output": outputs}

#TODO-Instance Segmentation Endpoint


if __name__ == "__main__":
    run("model_server:app", host="0.0.0.0", port=8000, reload=True)