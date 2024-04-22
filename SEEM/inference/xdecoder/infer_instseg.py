# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import logging

#Root path to the project for dependencies
pth = '/'.join(sys.path[0].split('/')[:-2])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(2)
#newline
import torch
from torchvision import transforms

from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from modeling.BaseModel import BaseModel
from modeling import build_model
from detectron2.utils.colormap import random_color
from utils.visualizer import Visualizer, ColorMode
from utils.distributed import init_distributed

logger = logging.getLogger(__name__)


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir

    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt['RESUME_FROM'])
    output_root = 'results/penguin_output'
    image_pth = 'SEEM/inference/images/penguin.jpeg'

    model = BaseModel(opt, build_model(opt)).from_pretrained(load_dir='SEEM/weights/seem_focall_v0.pt').eval().cuda()

    t = []
    t.append(transforms.Resize(800, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    thing_classes = ["penguin"]
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
        image_ori = Image.open(image_pth).convert('RGB')
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
        # Save the image
        summed_mask_image.save('results/penguin_output/summed_mask.png')

        if not os.path.exists(output_root):
            os.makedirs(output_root)
        demo.save(os.path.join(output_root, 'inst_rocks2.png'))

if __name__ == "__main__":
    main()
    sys.exit(0)