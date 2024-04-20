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




app = FastAPI()
args = parse_args()
# SEEM Model load args 
opt, cmdline_args = load_opt_command(args)
if cmdline_args.user_dir:
    absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
    opt['base_path'] = absolute_user_dir

# Pre trained Path
pretrained_pth =  'SEEM/weights/seem_focall_v0.pt'
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

#Declare the transform
t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)


class Item(BaseModel):
    image_pth : str
    thing_classes: List[str]
    stuff_classes: List[str]
    save_image: bool            #Save the image
    output_rool: str            #path to store results

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


        visual = Visualizer(image_ori, metadata=metadata)

        if item.save_image == True:
            pano_seg = outputs[-1]['panoptic_seg'][0]
            pano_seg_info = outputs[-1]['panoptic_seg'][1]

            for i in range(len(pano_seg_info)):
                if pano_seg_info[i]['category_id'] in metadata.thing_dataset_id_to_contiguous_id.keys():
                    pano_seg_info[i]['category_id'] = metadata.thing_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                else:
                    pano_seg_info[i]['isthing'] = False
                    pano_seg_info[i]['category_id'] = metadata.stuff_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]

            demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image

            if not os.path.exists(output_root):
                os.makedirs(output_root)
            demo.save(os.path.join(output_root, 'pano.png'))
    
    return {"output": outputs}

# def main(args = None):

#     # SEEM Model load args 
#     opt, cmdline_args = load_opt_command(args)
#     if cmdline_args.user_dir:
#         absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
#         opt['base_path'] = absolute_user_dir
    
#     # Pre trained Path
#     pretrained_pth =  'SEEM/weights/seem_focall_v0.pt'
#     model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

#     run(app, host="0.0.0.0", port = 8000)

# if __name__ == "__main__":
#     main()