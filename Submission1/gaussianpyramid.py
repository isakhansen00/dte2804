import cv2 as cv
import sys
from matplotlib import pyplot as plt
import os


img = cv.imread("sui.png")
lower = img.copy()

# Create a Gaussian Pyramid
gaussian_pyr = [lower]
for i in range(3):
   lower = cv.pyrDown(lower)
   gaussian_pyr.append(lower)

# Create windows for each layer of the Gaussian pyramid
# for i, layer in enumerate(gaussian_pyr):
#     cv.imshow(f'Gaussian Layer {i}', layer)

# Last level of Gaussian remains same in Laplacian
laplacian_top = gaussian_pyr[-1]
lp = [laplacian_top]

for i in range(3, 0, -1):
   size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
   gaussian_expanded = cv.pyrUp(gaussian_pyr[i], dstsize=size)
   laplacian = cv.subtract(gaussian_pyr[i-1], gaussian_expanded)
   #cv.imshow(str(i), laplacian)

upscaled_image = gaussian_pyr[-1]
for i in range(3):
   upscaled_image = cv.pyrUp(upscaled_image)

cv.imshow("Upscaled to original", upscaled_image)
cv.imwrite('test.png', gaussian_pyr[-1])
compressed_file_size = os.stat('test.png').st_size / 1000
print(f"{compressed_file_size:.0f}kB")
layer_3 = cv.pyrUp(gaussian_pyr[3])
# cv.imshow('Layer 3 Upscaled',layer_3)

# display all three layers
# cv.imshow('Layer 1',gaussian_pyr[3])
# cv.imshow('Layer 2',gaussian_pyr[2])
# cv.imshow("Layer 3", gaussian_pyr[1])
# cv.imshow("Original", gaussian_pyr[0])
cv.waitKey(0)