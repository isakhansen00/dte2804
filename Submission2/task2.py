import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# Function to calculate the Mean Absolute Difference (MAD) between two images
def calculate_mad(image1, image2):
    return np.mean(np.abs(image1 - image2))


# Load the image
image = cv2.imread('sui.png', cv2.IMREAD_GRAYSCALE)

# Define the block size
B = 8
def quantize(dct, quantize):
    quantized_dct_matrix = np.round(dct / quantize)
    return quantized_dct_matrix

def dequantize_dct(dct_matrix, quantization_matrix):
    # Reverse quantization by multiplying with the quantization matrix
    decompressed_dct_matrix = np.multiply(dct_matrix, quantization_matrix)
    
    return decompressed_dct_matrix
# Crop the image to ensure its dimensions are multiples of B
height, width = image.shape
new_height = (height // B) * B
new_width = (width // B) * B
cropped_image = image[:new_height, :new_width]
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
     [100, 100, 100, 90, 80, 20, 20, 20],
     [100, 100, 100, 100, 100, 20, 100, 2],
     [100, 100, 100, 90, 80, 80, 20, 40],
     [100, 100, 100, 100, 80, 100, 100, 40],
     [100, 100, 100, 100, 100, 100, 20, 10],
     [70, 80, 100, 100, 100, 20, 2, 10],
     [20, 100, 100, 100, 100, 100, 3, 2],
     [2, 100, 100, 90, 70, 20, 2, 2]
 ])

# c) Setting high and low-frequency parts to 0 and keeping the rest
quantization_matrix_c = np.array([
    [200, 200, 200, 200, 200, 200, 200, 200],
    [200, 200, 200, 200, 200, 200, 200, 200],
    [200, 200, 200, 200, 200, 200, 200, 200],
    [200, 200, 200, 200, 200, 200, 200, 200],
    [200, 200, 200, 200, 200, 200, 200, 200],
    [200, 200, 200, 200, 200, 200, 200, 200],
    [200, 200, 200, 200, 200, 200, 200, 200],
    [200, 200, 200, 200, 200, 200, 200, 200]
])
# Initialize variables to store the transformed and reconstructed images
Trans = np.zeros(cropped_image.shape, dtype=np.float32)
back0_a = np.zeros(cropped_image.shape, dtype=np.float32)
back0_b = np.zeros(cropped_image.shape, dtype=np.float32)
back0_c = np.zeros(cropped_image.shape, dtype=np.float32)

# Perform the DCT and IDCT block by block
for y in range(0, new_height, B):
    for x in range(0, new_width, B):
        block = cropped_image[y:y + B, x:x + B]

        # Apply DCT
        dct_block = cv2.dct(np.float32(block))
        Trans[y:y + B, x:x + B] = dct_block
        quantized_a = quantize(dct_block, quantization_matrix_a)
        quantized_b = quantize(dct_block, quantization_matrix_b)
        quantized_c = quantize(dct_block, quantization_matrix_c)

        dequantized_matrix_a = dequantize_dct(quantized_a, quantization_matrix_a)
        dequantized_matrix_b = dequantize_dct(quantized_b, quantization_matrix_b)
        dequantized_matrix_c = dequantize_dct(quantized_c, quantization_matrix_c)

        # Apply IDCT
        idct_block_a = cv2.idct(dequantized_matrix_a)
        idct_block_b = cv2.idct(dequantized_matrix_b)
        idct_block_c = cv2.idct(dequantized_matrix_c)
        back0_a[y:y + B, x:x + B] = idct_block_a
        back0_b[y:y + B, x:x + B] = idct_block_b
        back0_c[y:y + B, x:x + B] = idct_block_c

# Create a list to hold both the original and compressed images
images = []

images.append(image)
images.append(back0_a)
images.append(back0_b)
images.append(back0_c)

titles = ['Original', "A", "B", "C"]

# Create a 2x4 grid of subplots for displaying the images
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Iterate through subplots for displaying the images
for i in range(4):
    #axes[i].imshow(images[i], 'gray')  # Display the image in grayscale
    axes[i].imshow(images[i], cmap='gray', norm=Normalize(0, 255))
    axes[i].set_title(titles[i])  # Set the title for the subplot
    axes[i].set_xticks([])  # Remove the numbers on the x-axis
    axes[i].set_yticks([])  # Remove the numbers on the y-axis

# Adjust the horizontal spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Display the subplots
plt.show()