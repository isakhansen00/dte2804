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
# a)
sample_quantization_matrix_1 = np.array([
    [3, 5, 7, 9, 11, 13, 15, 17],
    [5, 7, 9, 11, 13, 15, 17, 19],
    [7, 9, 11, 13, 15, 17, 19, 21],
    [9, 11, 13, 15, 17, 19, 21, 23],
    [11, 13, 15, 17, 19, 21, 23, 25],
    [13, 15, 17, 19, 21, 23, 25, 27],
    [15, 17, 19, 21, 23, 25, 27, 29],
    [17, 19, 21, 23, 25, 27, 29, 31]
])

# b)
sample_quantization_matrix_2 = np.array([
    [100, 100, 100, 90, 80, 20, 20, 20],
    [100, 100, 100, 100, 100, 20, 100, 2],
    [100, 100, 100, 90, 80, 80, 20, 40],
    [100, 100, 100, 100, 80, 100, 100, 40],
    [100, 100, 100, 100, 100, 100, 20, 10],
    [70, 80, 100, 100, 100, 20, 2, 10],
    [20, 100, 100, 100, 100, 100, 3, 2],
    [2, 100, 100, 90, 70, 20, 2, 2]
])

# Step 2: Apply quantization to the DCT coefficients
quantized_image_1 = np.round(dct_image / sample_quantization_matrix_1)
quantized_image_2 = np.round(dct_image / sample_quantization_matrix_2)

# Create a list to hold both the original and compressed images
images = []

# Step 3: Inverse DCT to obtain compressed images
compressed_image_1 = scipy.fftpack.idct(scipy.fftpack.idct(quantized_image_1, axis=0, norm='ortho'), axis=1, norm='ortho')
compressed_image_2 = scipy.fftpack.idct(scipy.fftpack.idct(quantized_image_2, axis=0, norm='ortho'), axis=1, norm='ortho')

images.append(image)
images.append(compressed_image_1)
images.append(compressed_image_2)

titles = ['Original', "Sample Quantization Matrix", "Sample Quantization Matrix 2"]

# Create a 2x4 grid of subplots for displaying the images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Iterate through subplots for displaying the images
for i in range(3):
    axes[i].imshow(images[i], 'gray')  # Display the image in grayscale
    axes[i].set_title(titles[i])  # Set the title for the subplot
    axes[i].set_xticks([])  # Remove the numbers on the x-axis
    axes[i].set_yticks([])  # Remove the numbers on the y-axis

# Adjust the horizontal spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Display the subplots
plt.show()