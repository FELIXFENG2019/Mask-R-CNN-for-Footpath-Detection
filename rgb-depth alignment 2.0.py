import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import os


def depth2hue(depth_arr):
    d_min = 600
    d_max = 10000
    [rows, cols] = depth_arr.shape
    hue_image = np.zeros([rows, cols, 3])
    for i in range(0, rows):
        for j in range(0, cols):
            d = depth_arr[i, j]
            if d < d_min or d > d_max:
                hue_image[i, j, :] = 0
            else:
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
    hue_image = np.array(hue_image, dtype='uint8')
    return hue_image


def depth_normalization(depth_arr):
    d_min = 600
    d_max = 10000
    [rows, cols] = depth_arr.shape
    for i in range(0, rows):
        for j in range(0, cols):
            d = depth_arr[i, j]
            if d < d_min or d > d_max:
                depth_arr[i, j] = 0
    depth_arr = np.array(depth_arr, dtype='uint16')
    return depth_arr


print("Environment Ready")
# File names and paths initialization
frame_number = 900
image_number = "19"
bag_filename = "20200707_160533"

basic_name = bag_filename+'_'+image_number
bag_basepath = r"F:\Uob Dissertation Dataset\rosbag files"
bag_dir = os.path.join(bag_basepath, bag_filename+'.bag')
savepath_root = r"F:\Uob Dissertation Dataset\Aligned footpath dataset_new"
savepath = os.path.join(savepath_root, bag_filename)

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(bag_dir)
profile = pipe.start(cfg)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(frame_number):
    pipe.wait_for_frames()

# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()

# Cleanup:
pipe.stop()
print("Frames Captured")

# Original color frame
color = np.asanyarray(color_frame.get_data())
plt.rcParams["axes.grid"] = False
plt.rcParams['figure.figsize'] = [12, 6]
plt.imshow(color)
# plt.show()

# Original colirized depth frame
colorized_depth_arr = np.asanyarray(depth_frame.get_data())
colorized_depth = depth2hue(colorized_depth_arr)
plt.imshow(colorized_depth)
# plt.show()
images = np.hstack((color, colorized_depth))
plt.imshow(images)
plt.show()

# Create alignment primitive with color as its target stream:
align = rs.align(rs.stream.color)
frameset = align.process(frameset)

# Update color and depth frames:
aligned_depth_frame = frameset.get_depth_frame()
aligned_depth = np.asanyarray(aligned_depth_frame.get_data())
aligned_depth = depth_normalization(aligned_depth)
# print(aligned_depth)
# print(aligned_depth.shape)
# Save the normalized depth data as npy
depth_filename = basic_name+'_depth.npy'
temp_dir = os.path.join(savepath, depth_filename)
# np.save(temp_dir, aligned_depth)

# colorized_depth_arr = np.asanyarray(aligned_depth_frame.get_data())
colorized_depth = depth2hue(aligned_depth)

# Show the two frames together:
images = np.hstack((color, colorized_depth))
plt.imshow(images)
plt.show()

# plt.imshow(color)
# plt.show()
color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
temp_dir = os.path.join(savepath, basic_name+'_color.png')
# cv2.imwrite(temp_dir, color_bgr)

# plt.imshow(colorized_depth)
# plt.show()
colorized_depth_bgr = cv2.cvtColor(colorized_depth, cv2.COLOR_RGB2BGR)
temp_dir = os.path.join(savepath, basic_name+'_depth_visualization.png')
# cv2.imwrite(temp_dir, colorized_depth_bgr)

