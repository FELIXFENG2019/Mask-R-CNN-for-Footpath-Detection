import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

image_arr = cv2.imread(r'G:\DIODE Dataset\scene_00007\scan_00084\00007_00084_outdoor_000_000.png')
image_arr_rgb = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
print(image_arr_rgb.shape)
plt.imshow(image_arr_rgb)
plt.show()

arr = np.load(r'G:\DIODE Dataset\scene_00007\scan_00084\00007_00084_outdoor_000_000_depth.npy')
#arr = np.load(r"C:\Users\FENGSHIJIA\Desktop\aligned_depth_1.npy")
print(arr.shape)
print(arr)
arr = np.reshape(arr, (768, 1024))
print(arr.shape)
print(arr)


arr_mask = np.load(r'G:\DIODE Dataset\scene_00007\scan_00084\00007_00084_outdoor_000_000_depth_mask.npy')
print(arr_mask.shape)
print(arr_mask)

