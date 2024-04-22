## Write a module that start up both the SEEM and LLaVA models together, we would like to use LLaVA to 
## generate a prompt for the SEEM model.

import sys
import os

# Add the SEEM path to the system path
pth = '/home/ommos92/adv-computer-vision/SVLM/SEEM/'
sys.path.insert(0, pth)

# Add the LLaVA path to the system path
pth = '/home/ommos92/adv-computer-vision/SVLM/LLaVA'
sys.path.insert(0, pth)

from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.eval.grounded_llava import ground_model

model_path = "liuhaotian/llava-v1.5-7b"
prompt = "List all of the objects."
image_file = "SEEM/inference/images/penguin.jpeg"

llava_args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

results = ground_model(llava_args)
print(results)



