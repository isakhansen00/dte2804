import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
import cv2

# Load an example image
image_path = "sui.png"
n=2
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

coeffs = pywt.wavedec2(original_image, 'haar', level=n)

coeffs[0] /= np.abs(coeffs[0]).max()
for detail_level in range(n):
    coeffs[detail_level + 1] = [d/np.abs(d).max() for d in 
                                coeffs[detail_level + 1]]
    
array, coeff_slices = pywt.coeffs_to_array(coeffs)
plt.imshow(array, cmap='gray', vmin=-0.25, vmax=0.75)
plt.show()
