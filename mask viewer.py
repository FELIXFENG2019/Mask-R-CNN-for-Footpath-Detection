import numpy as np
import matplotlib.pyplot as plt
import cv2


mask_path = r"F:\Uob Dissertation Dataset\footpath dataset 2.0\val\20200707_155024_2_mask.npy"
mask = np.load(mask_path)
height, width, depth = mask.shape
# print(depth)

image_arr = np.zeros([height, width])
for i in range(0, depth):
    if i == 0:
        x = 20
    else:
        x = (i - 0) / depth * 255
    image_arr = x * mask[:, :, i] + image_arr
    mask_image = image_arr

mask_image = np.float32(mask_image)
mask_image_RGB = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
plt.imshow(mask_image_RGB)
plt.show()
plt.imshow(mask_image)
plt.show()

