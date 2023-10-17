import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import cv2

# Load your image
# Define an 8x8 grayscale image
image = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73],
    [63, 59, 55, 90, 109, 85, 69, 72],
    [62, 59, 68, 113, 144, 104, 66, 73],
    [63, 58, 71, 122, 154, 106, 70, 69],
    [67, 61, 68, 104, 126, 88, 68, 70],
    [79, 65, 60, 70, 77, 68, 58, 75],
    [85, 71, 64, 59, 55, 61, 65, 83],
    [87, 79, 69, 68, 65, 76, 78, 94]
], dtype=np.float32)

# Step 1: Perform 2D DCT on the image
dct_image = scipy.fftpack.dct(scipy.fftpack.dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

# Quantization matrices for the three scenarios
# a) Normal Compression (High-frequency parts set to 0)
quantization_matrix_a = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# b) Setting only low-frequency parts to 0
quantization_matrix_b = np.array([
    [128, 128, 128, 128, 128, 128, 128, 128],
    [128, 128, 128, 128, 128, 128, 128, 128],
    [128, 128, 128, 128, 128, 128, 128, 128],
    [128, 128, 128, 128, 128, 128, 128, 128],
    [128, 128, 128, 128, 128, 128, 128, 128],
    [128, 128, 128, 128, 128, 128, 128, 128],
    [128, 128, 128, 128, 128, 128, 128, 128],
    [128, 128, 128, 128, 128, 128, 128, 128]
])

# c) Setting high and low-frequency parts to 0 and keeping the rest
quantization_matrix_c = np.array([
    [16, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

# Step 2: Apply quantization to the DCT coefficients
quantized_image_a = np.round(dct_image / quantization_matrix_a)
quantized_image_b = np.round(dct_image / quantization_matrix_b)
quantized_image_c = np.round(dct_image / quantization_matrix_c)

# Create a list to hold both the original and compressed images
images = []

# Step 3: Inverse DCT to obtain compressed images
compressed_image_a = scipy.fftpack.idct(scipy.fftpack.idct(quantized_image_a, axis=0, norm='ortho'), axis=1, norm='ortho')
compressed_image_b = scipy.fftpack.idct(scipy.fftpack.idct(quantized_image_b, axis=0, norm='ortho'), axis=1, norm='ortho')
compressed_image_c = scipy.fftpack.idct(scipy.fftpack.idct(quantized_image_c, axis=0, norm='ortho'), axis=1, norm='ortho')

images.append(image)
images.append(compressed_image_a)
images.append(compressed_image_b)
images.append(compressed_image_c)

titles = ['Original', "A", "B", "C"]

# Create a 2x4 grid of subplots for displaying the images
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Iterate through subplots for displaying the images
for i in range(4):
    axes[i].imshow(images[i], 'gray')  # Display the image in grayscale
    axes[i].set_title(titles[i])  # Set the title for the subplot
    axes[i].set_xticks([])  # Remove the numbers on the x-axis
    axes[i].set_yticks([])  # Remove the numbers on the y-axis

# Adjust the horizontal spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Display the subplots
plt.show()