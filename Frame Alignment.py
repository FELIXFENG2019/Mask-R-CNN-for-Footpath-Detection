import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API


color = cv2.imread(r"C:\Users\FENGSHIJIA\Desktop\1 rgb-d\1.1_Color.png")
#color = np.asanyarray(color_frame.get_data())
print(color.shape)
print(color)
plt.imshow(color)
plt.show()

colorizer = rs.colorizer()
colorized_depth = cv2.imread(r"C:\Users\FENGSHIJIA\Desktop\1 rgb-d\1.2_Depth.png")
print(colorized_depth.shape)
print(colorized_depth)
plt.imshow(colorized_depth)
plt.show()

# Create alignment primitive with color as its target stream:
align = rs.align(rs.stream.color)
frameset = align.process(frameset)

# Update color and depth frames:
aligned_depth_frame = frameset.get_depth_frame()
colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

# Show the two frames together:
images = np.hstack((color, colorized_depth))
print(color)
print(colorized_depth)
plt.imshow(images)
plt.show()
