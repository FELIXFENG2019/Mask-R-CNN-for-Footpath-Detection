import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API

print("Environment Ready")

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(r"F:\Uob Dissertation Dataset\rosbag files\20200707_155024.bag")
profile = pipe.start(cfg)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(65):
    pipe.wait_for_frames()

# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()

# Cleanup:
pipe.stop()
print("Frames Captured")

color = np.asanyarray(color_frame.get_data())
plt.rcParams["axes.grid"] = False
plt.rcParams['figure.figsize'] = [12, 6]
plt.imshow(color)
plt.show()

colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 9)  # 0-Jet, 9-Hue
colorizer.set_option(rs.option.min_distance, 0.1000)
colorizer.set_option(rs.option.min_distance, 10.0000)
colorizer.set_option(rs.option.histogram_equalization_enabled, 0)

colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
plt.imshow(colorized_depth)
plt.show()

# Create alignment primitive with color as its target stream:
align = rs.align(rs.stream.color)
frameset = align.process(frameset)

# Update color and depth frames:
aligned_depth_frame = frameset.get_depth_frame()
aligned_depth = np.asanyarray(aligned_depth_frame.get_data())
print(aligned_depth)
print(aligned_depth.shape)
np.save(r"C:\Users\FENGSHIJIA\Desktop\aligned_depth_1.npy", aligned_depth)

colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

# Show the two frames together:
images = np.hstack((color, colorized_depth))
plt.imshow(images)
plt.show()

# plt.imshow(color)
# plt.show()
color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
# cv2.imwrite(r"F:\Uob Dissertation Dataset\Aligned footpath dataset\20200707_161711\aligned_color_18.png", color_bgr)

# plt.imshow(colorized_depth)
# plt.show()
colorized_depth_bgr = cv2.cvtColor(colorized_depth, cv2.COLOR_RGB2BGR)
# cv2.imwrite(r"F:\Uob Dissertation Dataset\Aligned footpath dataset\20200707_161711\aligned_depth_18.png", colorized_depth_bgr)
