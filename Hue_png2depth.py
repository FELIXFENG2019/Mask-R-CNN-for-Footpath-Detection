import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

arr = cv2.imread(r"G:\SUNRGBD\realsense\sh\2014_10_21-11_35_34-1311000041\fullres\0000067.png", cv2.IMREAD_GRAYSCALE)
print(arr)
print(arr.shape)
