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


figure, axs = plt.subplots(4, 2, figsize=(10, 15))
        
# Create windows for each layer of the Gaussian pyramid
for i, layer in enumerate(gaussian_pyr):
    if i == 0:
        axs[i, 0].imshow(cv.cvtColor(layer, cv.COLOR_BGR2RGB))
        axs[i, 0].set_title(f'Original image')
        axs[i, 0].axis('off')

    else:   
        axs[i, 0].imshow(cv.cvtColor(layer, cv.COLOR_BGR2RGB))
        axs[i, 0].set_title(f'Gaussian layer: {i}')
        axs[i, 0].axis('off')


# Last level of Gaussian remains same in Laplacian
laplacian_top = gaussian_pyr[-1]
lp = [laplacian_top]

for i in range(3, 0, -1):
   size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
   gaussian_expanded = cv.pyrUp(gaussian_pyr[i], dstsize=size)
   laplacian = cv.subtract(gaussian_pyr[i-1], gaussian_expanded)
   axs[i - 1, 1].imshow(cv.cvtColor(laplacian, cv.COLOR_BGR2RGB))
   axs[i - 1, 1].set_title(f'Laplacian Layer {i}')
   axs[i - 1, 1].axis('off')

# Set the title for the last Laplacian layer
lp.append(gaussian_pyr[0])
axs[0, 1].set_title('Laplacian Layer 0')

# Remove any remaining empty subplot(s)
if len(gaussian_pyr) < 3:
    axs[-1, 0].axis('off')

# Show the plot
#plt.tight_layout()
plt.show()


# Code in block comment below is used for saving the compressed image and then reading out its new file size
'''
upscaled_image = gaussian_pyr[-1]
for i in range(3):
    upscaled_image = cv.pyrUp(upscaled_image)

cv.imshow("Upscaled to original", upscaled_image)
cv.imwrite('test.png', gaussian_pyr[-1])
compressed_file_size = os.stat('test.png').st_size / 1000
print(f"{compressed_file_size:.0f}kB")
'''

# # display all three layers
# # cv.imshow('Layer 1',gaussian_pyr[3])
# # cv.imshow('Layer 2',gaussian_pyr[2])
# # cv.imshow("Layer 3", gaussian_pyr[1])
# # cv.imshow("Original", gaussian_pyr[0])
# cv.waitKey(0)