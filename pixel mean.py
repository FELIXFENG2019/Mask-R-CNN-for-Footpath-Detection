import os
import numpy as np
import fnmatch
import cv2
from tqdm import tqdm

basepath = r"F:\Uob Dissertation Dataset\footpath dataset_splitted"
r_means = []
g_means = []
b_means = []
images = []

subsets = ['train', 'val', 'test']
for subset in subsets:
    level1_dir = os.path.join(basepath, subset)
    files = os.listdir(level1_dir)
    for file in files:
        if fnmatch.fnmatch(file, '*_depth_visualization.png'):
            images.append(file)

    with tqdm(total=len(images), desc=subset, leave=True, unit='img', unit_scale=True) as pbar:
        for image in images:
            img_path = os.path.join(level1_dir, image)
            image_arr = cv2.imread(img_path)
            image_arr_b = image_arr[:, :, 0]
            image_arr_g = image_arr[:, :, 1]
            image_arr_r = image_arr[:, :, 2]
            per_img_b_mean = np.mean(image_arr_b)
            per_img_g_mean = np.mean(image_arr_g)
            per_img_r_mean = np.mean(image_arr_r)
            b_means.append(per_img_b_mean)
            g_means.append(per_img_g_mean)
            r_means.append(per_img_r_mean)

            pbar.update(1)

    images = []

B_mean = np.mean(b_means)
G_mean = np.mean(g_means)
R_mean = np.mean(r_means)

B_std = np.std(b_means)
G_std = np.std(g_means)
R_std = np.std(r_means)

print("Calculation completed!")

print('R mean=', R_mean, 'G mean=', G_mean, 'B mean=', B_mean)
print('R std=', R_std, 'G std=', G_std, 'B std=', B_std)

