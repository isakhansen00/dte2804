import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
original_image = cv2.imread('minions.jpg')
original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Define the region of interest (ROI) - for example, top-left corner
x, y, width, height = 0, 0, 300, 150  # Adjust these values as needed
roi = original_image_gray[y:y+height, x:x+width]

# Apply the DCT to the ROI
dct_roi = cv2.dct(np.float32(roi))

# Perform the inverse DCT on the DCT coefficients
roi_reconstructed = cv2.idct(dct_roi)

# Display the images
plt.figure(figsize=(12, 8))

plt.subplot(231), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(232), plt.imshow(original_image_gray, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(233), plt.imshow(roi, cmap='gray')
plt.title('ROI')

plt.subplot(234), plt.imshow(np.log(np.abs(dct_roi)), cmap='gray')
plt.title('DCT Coefficients (ROI)')

plt.subplot(235), plt.imshow(roi_reconstructed, cmap='gray')
plt.title('Inverse DCT (ROI)')

plt.tight_layout()
plt.show()
