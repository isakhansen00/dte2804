import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import scipy.fftpack as fft
from dahuffman import HuffmanCodec
import sys


# Function to calculate the Mean Absolute Difference (MAD) between two images
def calculate_mad(image1, image2):
    return np.mean(np.abs(image1 - image2))

# Load the image
image = cv2.imread('sui.png', cv2.IMREAD_GRAYSCALE)

# Define the block size
B = 8
def quantize(fft, quantize):
    quantized_fft_matrix = np.round(fft / quantize)
    return quantized_fft_matrix

def dequantize_fft(fft_matrix, quantization_matrix):
    # Reverse quantization by multiplying with the quantization matrix
    decompressed_fft_matrix = np.multiply(fft_matrix, quantization_matrix)
    return decompressed_fft_matrix

# Huffman encoding and decoding functions
def huffman_encode(data):
    codec = HuffmanCodec.from_data(data)
    encoded = codec.encode(data)
    return encoded, codec

def huffman_decode(encoded, codec):
    decoded = codec.decode(encoded)
    return decoded

# Crop the image to ensure its dimensions are multiples of B
height, width = image.shape
new_height = (height // B) * B
new_width = (width // B) * B
cropped_image = image[:new_height, :new_width]

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
# Initialize variables to store the transformed and reconstructed images
Trans = np.zeros(cropped_image.shape, dtype=np.float32)
back0_a = np.zeros(cropped_image.shape, dtype=np.float32)

# Perform the FFT and IFFT block by block
for y in range(0, new_height, B):
    for x in range(0, new_width, B):
        block = cropped_image[y:y + B, x:x + B]

        # Apply FFT
        fft_block = fft.fft(np.float32(block))
        Trans[y:y + B, x:x + B] = fft_block
        # Apply quantization
        quantized_a = quantize(fft_block, quantization_matrix)

        # Convert the quantized data to a 1D array (you may need to reshape or flatten it)
        quantized_a_1d = quantized_a.reshape(-1)
        
        # Apply Huffman encoding
        encoded_data, huffman_codec = huffman_encode(quantized_a_1d)
        
        # Apply Huffman decoding
        decoded_data = huffman_decode(encoded_data, huffman_codec)

        # Convert the decoded data list to a NumPy array
        decoded_block = np.array(decoded_data).reshape(B, B)

        # Apply dequantization
        dequantized_matrix_a = dequantize_fft(decoded_block, quantization_matrix)
        # Apply IFFT
        ifft_block_a = fft.ifft(dequantized_matrix_a)
        back0_a[y:y + B, x:x + B] = ifft_block_a

# Create a list to hold both the original and compressed images
images = []

images.append(image)
images.append(back0_a)

titles = ['Original', "A", "B", "C"]

# Create a 2x4 grid of subplots for displaying the images
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Iterate through subplots for displaying the images
for i in range(2):
    #axes[i].imshow(images[i], 'gray')  # Display the image in grayscale
    axes[i].imshow(images[i], cmap='gray', norm=Normalize(0, 255))
    axes[i].set_title(titles[i])  # Set the title for the subplot
    axes[i].set_xticks([])  # Remove the numbers on the x-axis
    axes[i].set_yticks([])  # Remove the numbers on the y-axis

# Adjust the horizontal spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Display the subplots
plt.show()

# Calculate the size in bytes of the original image
original_image_size = sys.getsizeof(image)

# Calculate the size in bytes of the encoded data
encoded_data_size = sys.getsizeof(encoded_data)

# Calculate the size in bytes of the decoded data
decoded_data_size = sys.getsizeof(decoded_data)

# Calculate the compression ratio
compression_ratio = original_image_size / encoded_data_size

print(f"Original image size: {original_image_size} bytes")
print(f"Encoded data size: {encoded_data_size} bytes")
print(f"Decoded data size: {decoded_data_size} bytes")
print(f"Compression Ratio: {compression_ratio:.2f}")