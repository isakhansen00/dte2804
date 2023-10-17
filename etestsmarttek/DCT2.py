import cv2
import numpy as np

def pad_block(block, target_shape=(8, 8)):
    padded_block = np.zeros(target_shape)
    padded_block[:block.shape[0], :block.shape[1]] = block
    return padded_block

def apply_dct_and_quantization(img_block, quantization_matrix):
    # Perform DCT on the block
    dct_matrix = cv2.dct(np.float32(img_block))
    
    # Quantize the DCT coefficients
    quantized_dct_matrix = np.round(dct_matrix / quantization_matrix)
    
    return quantized_dct_matrix

def dequantize_dct(quantized_dct_matrix, quantization_matrix):
    # Reverse quantization by multiplying with the quantization matrix
    decompressed_dct_matrix = quantized_dct_matrix * quantization_matrix
    
    return decompressed_dct_matrix

# Define the standard JPEG quantization matrix
quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Read the original image
img = cv2.imread("sui.png", cv2.IMREAD_GRAYSCALE)

# Get the size of the original image
height, width = img.shape

# Define block size
block_size = 8

# Process the image in 8x8 blocks
for i in range(0, height, block_size):
    for j in range(0, width, block_size):
        # Extract an 8x8 block from the original image
        img_block = img[i:i+block_size, j:j+block_size]
        
        # Pad the block if its size is smaller than 8x8
        padded_block = pad_block(img_block)
        
        # Apply DCT and quantization to the block
        quantized_dct_matrix = apply_dct_and_quantization(padded_block, quantization_matrix)
        
        # Dequantize the block
        decompressed_dct_matrix = dequantize_dct(quantized_dct_matrix, quantization_matrix)
        
        # Perform inverse DCT to obtain the block
        reconstructed_block = cv2.idct(np.float32(decompressed_dct_matrix))
        
        # Place the reconstructed block back into the image
        img[i:i+block_size, j:j+block_size] = reconstructed_block

# Clip the values to be within the valid pixel range [0, 255]
reconstructed_image = np.clip(img, 0, 255)

# Convert back to uint8 data type
reconstructed_image = np.uint8(reconstructed_image)

# Display or save the reconstructed image
cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

