import numpy as np
import requests
import json

# Define the URL of the API endpoint
url = "http://localhost:8000/process_image"

# Define the numpy array you want to send
# Numpy of size 900,1500 with all white pixels
mask_image = np.ones((900, 1500))

# Convert the numpy array to a list
mask_image_list = mask_image.tolist()

# Define the image path you want to send
image_path = "SEEM/inference/images/penguin.jpeg"

# Define the prompt you want to send
prompt = "Personify the penguins in the image."

# Make the POST request
response = requests.post(
    url,
    json={
        "image_path": image_path,
        "mask_image": mask_image_list,
        "prompt": prompt,
    },
)

# Print the response
print(response.status_code)