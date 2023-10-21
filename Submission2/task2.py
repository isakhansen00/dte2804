import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

# Function to quantize the DCT coefficients
def quantize(dct, quantize):
    quantized_dct_matrix = np.round(dct / quantize)
    return quantized_dct_matrix

# Function to dequantize the DCT coefficients
def dequantize_dct(dct_matrix, quantization_matrix):
    # Reverse quantization by multiplying with the quantization matrix
    decompressed_dct_matrix = np.multiply(dct_matrix, quantization_matrix)
    return decompressed_dct_matrix

# Function to apply DCT and quantization to image blocks
def dtc_quantiazation_pipeline(image, B, quantization_matrix_a, quantization_matrix_b, quantization_matrix_c):

    # Crop the image to ensure its dimensions are multiples of B
    height, width = image.shape
    new_height = (height // B) * B
    new_width = (width // B) * B
    cropped_image = image[:new_height, :new_width]


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

            # Apply quantization using the three different matrices
            quantized_a = quantize(dct_block, quantization_matrix_a)
            quantized_b = quantize(dct_block, quantization_matrix_b)
            quantized_c = quantize(dct_block, quantization_matrix_c)

            # Apply dequantization using the three different matrices
            dequantized_matrix_a = dequantize_dct(quantized_a, quantization_matrix_a)
            dequantized_matrix_b = dequantize_dct(quantized_b, quantization_matrix_b)
            dequantized_matrix_c = dequantize_dct(quantized_c, quantization_matrix_c)

            # Apply IDCT to reconstruct the image
            idct_block_a = cv2.idct(dequantized_matrix_a)
            idct_block_b = cv2.idct(dequantized_matrix_b)
            idct_block_c = cv2.idct(dequantized_matrix_c)

            # Store the reconstructed images
            back0_a[y:y + B, x:x + B] = idct_block_a
            back0_b[y:y + B, x:x + B] = idct_block_b
            back0_c[y:y + B, x:x + B] = idct_block_c

    # Create a list to hold both the original and compressed images
    images = []

    images.append(image)  # Original image
    images.append(back0_a)  # Compressed image with high-frequency parts to 0
    images.append(back0_b)  # Compressed image with low-frequency parts to 0
    images.append(back0_c)  # Compressed image with both high and low-frequency parts to 0

    return images

# Function to plot the resulting images
def plot_result(images):
    titles = ['Original', "High-frequency parts to 0", "Low-frequency parts to 0", "Both high and low-frequency parts to 0", 'Original', "High-frequency parts to 0", "Low-frequency parts to 0", "Both high and low-frequency parts to 0"]

    # Create a 2x4 grid of subplots for displaying the images
    fig, axes = plt.subplots(2, 4, figsize=(10, 10))

    # Iterate through subplots for displaying the images
    for i in range(8):
        row = i // 4  # Determine the row
        col = i % 4   # Determine the column
        axes[row, col].imshow(images[i], cmap='gray', norm=Normalize(0, 255))
        axes[row, col].set_title(titles[i])
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

    # Adjust the horizontal and vertical spacing between subplots
    plt.subplots_adjust(wspace=0.5)

    # Display the subplots
    plt.show()

def main():

    # List for all original images
    orignial_images = []

    # List for all images who are going to be plotted
    plotting_images = []

    # Load the image
    orignial_images.append(cv2.imread('sui.png', cv2.IMREAD_GRAYSCALE))
    orignial_images.append(cv2.imread('messi.jpeg', cv2.IMREAD_GRAYSCALE))

    # Define the block size
    B = 8

    # a) Setting only high-frequency parts to 0
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

    # Loop through each original image and run it through the pipelie
    for image in orignial_images:
        images = dtc_quantiazation_pipeline(image, B, quantization_matrix_a, quantization_matrix_b, quantization_matrix_c)
        plotting_images = plotting_images + images
    
    # Plotting results
    plot_result(plotting_images)

if __name__ == "__main__":
    main()
