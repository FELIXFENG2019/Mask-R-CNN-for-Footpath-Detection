import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def depth2hue(depth_arr):
    d_min = 100
    d_max = 10000
    disp_min = 1 / d_max
    disp_max = 1 / d_min
    [rows, cols] = depth_arr.shape
    hue_image = np.zeros([rows, cols, 3])
    for i in range(0, rows):
        for j in range(0, cols):
            d = depth_arr[i, j]
            # disp = 1 / d
            if d <= d_min or d >= d_max:
                hue_image[i, j, :] = 0
            else:
                # d_normal = 1529 * (disp - disp_min) / (disp_max - disp_min)
                d_normal = 1529 * (d - d_min) / (d_max - d_min)
                # print(d_normal)
                # pr
                if (0 <= d_normal <= 255) or (1257 < d_normal <= 1529):
                    hue_image[i, j, 0] = 255
                elif 255 < d_normal <= 510:
                    hue_image[i, j, 0] = 510 - d_normal
                elif 510 < d_normal <= 1020:
                    hue_image[i, j, 0] = 0
                elif 1020 < d_normal <= 1275:
                    hue_image[i, j, 0] = d_normal - 1020
                # pg
                if 0 < d_normal <= 255:
                    hue_image[i, j, 1] = d_normal
                elif 255 < d_normal <= 510:
                    hue_image[i, j, 1] = 255
                elif 510 < d_normal <= 765:
                    hue_image[i, j, 1] = 255
                elif 765 < d_normal <= 1020:
                    hue_image[i, j, 1] = 1020 - d_normal
                elif 1020 < d_normal <= 1529:
                    hue_image[i, j, 1] = 0
                # pb
                if 0 < d_normal <= 510:
                    hue_image[i, j, 2] = 0
                elif 510 < d_normal <= 765:
                    hue_image[i, j, 2] = d_normal - 510
                elif 765 < d_normal <= 1020:
                    hue_image[i, j, 2] = 255
                elif 1020 < d_normal <= 1275:
                    hue_image[i, j, 2] = 255
                elif 1275 < d_normal <= 1529:
                    hue_image[i, j, 2] = 1530 - d_normal
    hue_image = hue_image.astype(int)
    return hue_image


filename = r'F:\Uob Dissertation Dataset\footpath dataset original captured\1_20200707_155024\1_Depth.raw'
depth_image = np.fromfile(filename, dtype='uint16')
print(depth_image)
print(depth_image.shape)

depth_image = depth_image.reshape(480, 640)
print(depth_image)
print(depth_image.shape)

hue = depth2hue(depth_image)
print(hue)

#for i in range(0, 480):
    #for j in range(0, 640):
        #print(hue[i, j])

print(hue.shape)
plt.imshow(hue)
plt.show()

"""
arr = np.load(r'G:\DIODE Dataset\scene_00007\scan_00084\00007_00084_outdoor_000_000_depth.npy')
print(arr.shape)
arr = np.reshape(arr, (768, 1024))
arr = 1000 * arr
print(arr)
arr_hue = depth2hue(arr)
print(arr_hue)
print(arr_hue.shape)
plt.imshow(arr_hue)
plt.show()
"""

arr = np.load(r'C:\Users\FENGSHIJIA\Desktop\aligned_depth_1.npy')
print(arr.shape)
print(arr)
arr_hue = depth2hue(arr)
print(arr_hue)
print(arr_hue.shape)
plt.imshow(arr_hue)
plt.show()

