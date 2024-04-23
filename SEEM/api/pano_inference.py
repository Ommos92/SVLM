from typing import List

from fastapi import FastAPI, UploadFile, File

import os
import sys
import logging
import argparse

pth = '/home/ommos92/adv-computer-vision/SVLM/SEEM/'
sys.path.insert(0, pth)

from fastapi import FastAPI
from uvicorn import run
import pydantic 


import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.arguments import load_opt_command
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from modeling.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from modeling.BaseModel import BaseModel
from modeling import build_model

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


class Item(pydantic.BaseModel):
    image_pth : str = pydantic.Field(..., example='SEEM/inference/images/street.jpg')
    thing_classes: List[str] = pydantic.Field(..., example=['car','person','traffic light', 'truck', 'motorcycle'])
    stuff_classes: List[str] = pydantic.Field(..., example=['building','sky','street','tree','rock','sidewalk'])
    save_image: bool = pydantic.Field(..., example=True)            #Save the image
    output_root: str = pydantic.Field(..., example="path/to/output")


def predict_instseg(image_path = None, thing_classes = None, stuff_classes = None, save_image = True, output_root = None):

    thing_classes = thing_classes
    thing_colors = [random_color(rgb=True, maximum=255).astype(int).tolist() for _ in range(len(thing_classes))]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + ["background"], is_eval=False)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes)

    with torch.no_grad():
        image_ori = Image.open(image_path).convert('RGB')
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs)
        visual = Visualizer(image_ori, metadata=metadata)

        inst_seg = outputs[-1]['instances']
        inst_seg.pred_masks = inst_seg.pred_masks.cpu()
        inst_seg.pred_boxes = BitMasks(inst_seg.pred_masks > 0).get_bounding_boxes()
        
        #demo = visual.draw_instance_predictions(inst_seg) # rgb Image
        demo, masks = visual.mask_predictions(inst_seg)
        
        # Initialize an empty array of the same shape as the masks
        summed_mask = np.zeros_like(masks[0].mask)

        # Sum all masks for a singular mask
        for mask in masks:
            binary_mask = mask.mask.astype(np.uint8) * 255
            summed_mask += binary_mask

        # Convert the summed mask to an image
        summed_mask_image = Image.fromarray(summed_mask)
        
        output_root = "results"
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        demo.save(os.path.join(output_root, 'grounded_llava.png'))
        
    return {"output": outputs, "mask": summed_mask_image}