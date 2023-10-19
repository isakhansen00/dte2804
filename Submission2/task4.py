import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data

import cv2

# Load an example image
image_path = "sui.png"
n=2
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Perform 2D wavelet transform (MRA) on the original image
coeffs2 = pywt.dwt2(original_image, 'haar')
LL, (LH, HL, HH) = coeffs2

# Define meta information (for example, a watermark)
meta_info = np.random.randint(0, 128, size=LL.shape)  # Ensure meta_info has the same dimensions as LL




# Resize meta_info to match the shape of LL
meta_info_resized = cv2.resize(meta_info, (LL.shape[1], LL.shape[0]))

# Exchange the LL (approximation) coefficients with meta information
LL_with_meta_info = LL + meta_info_resized

# Reconstruct the image using the modified coefficients
modified_image = pywt.idwt2((LL_with_meta_info, (LH, HL, HH)), 'haar')

# Plot the original and modified images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(modified_image, cmap='gray')
plt.title('Modified Image with Meta Information')
plt.axis('off')

plt.tight_layout()
plt.show()