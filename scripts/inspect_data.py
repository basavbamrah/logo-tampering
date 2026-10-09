import torch 
import numpy
from PIL import Image
import cv2
import os

folder_path = "/home/darklord/Desktop/proj/Logo_Tampering/candidate_package"

img_sizes = []
for file in os.listdir(os.path.join(folder_path, "images")):
    image_path = os.path.join(folder_path, "images", file)
    image = Image.open(image_path)
    image = image.convert("RGB")
    img_sizes.append(image.size)

print(img_sizes)
print("Range of image sizes: ", min(img_sizes), max(img_sizes))

