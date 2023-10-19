import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import scipy.fftpack as fft
from dahuffman import HuffmanCodec
import sys

# Load the images
image_ronaldo = cv2.imread('sui.png', cv2.IMREAD_GRAYSCALE)
image_messi = cv2.imread('messi.jpeg', cv2.IMREAD_GRAYSCALE)

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
height, width = image_ronaldo.shape
new_height = (height // B) * B
new_width = (width // B) * B
cropped_image_ronaldo = image_ronaldo[:new_height, :new_width]

height2, width2 = image_messi.shape
new_height2 = (height2 // B) * B
new_width2 = (width2 // B) * B
cropped_image_messi = image_messi[:new_height2, :new_width2]

quantization_matrix = np.array([
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
trans_ronaldo = np.zeros(cropped_image_ronaldo.shape, dtype=np.float32)
trans_messi = np.zeros(cropped_image_messi.shape, dtype=np.float32)
reconstructed_ronaldo = np.zeros(cropped_image_ronaldo.shape, dtype=np.float32)
reconstructed_messi = np.zeros(cropped_image_messi.shape, dtype=np.float32)

# Perform the FFT and IFFT block by block (Ronaldo)
for y in range(0, new_height, B):
    for x in range(0, new_width, B):
        block = cropped_image_ronaldo[y:y + B, x:x + B]

        # Apply FFT
        fft_block = fft.fft(np.float32(block))
        trans_ronaldo[y:y + B, x:x + B] = fft_block

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
        reconstructed_ronaldo[y:y + B, x:x + B] = ifft_block_a

# Perform the FFT and IFFT block by block (Messi)
for y in range(0, new_height2, B):
    for x in range(0, new_width2, B):
        block2 = cropped_image_messi[y:y + B, x:x + B]

        # Apply FFT
        fft_block2 = fft.fft(np.float32(block2))
        trans_messi[y:y + B, x:x + B] = fft_block2

        # Apply quantization
        quantized_b = quantize(fft_block2, quantization_matrix)

        # Convert the quantized data to a 1D array (you may need to reshape or flatten it)
        quantized_b_1d = quantized_b.reshape(-1)
        
        # Apply Huffman encoding
        encoded_data2, huffman_codec2 = huffman_encode(quantized_b_1d)
        
        # Apply Huffman decoding
        decoded_data2 = huffman_decode(encoded_data2, huffman_codec2)

        # Convert the decoded data list to a NumPy array
        decoded_block2 = np.array(decoded_data2).reshape(B, B)

        # Apply dequantization
        dequantized_matrix_b = dequantize_fft(decoded_block2, quantization_matrix)

        # Apply IFFT
        ifft_block_b = fft.ifft(dequantized_matrix_b)
        reconstructed_messi[y:y + B, x:x + B] = ifft_block_b

# Calculate the size in bytes of the original image
original_image_size_ronaldo = sys.getsizeof(image_ronaldo)
original_image_size_messi = sys.getsizeof(image_messi)

# Calculate the size in bytes of the encoded data
encoded_data_size_ronaldo = sys.getsizeof(encoded_data)
encoded_data_size_messi = sys.getsizeof(encoded_data2)

# Calculate the size in bytes of the decoded data
decoded_data_size_ronaldo = sys.getsizeof(decoded_data)
decoded_data_size_messi = sys.getsizeof(decoded_data2)

# Calculate the compression ratio
compression_ratio_ronaldo = original_image_size_ronaldo / encoded_data_size_ronaldo
compression_ratio_messi = original_image_size_messi / encoded_data_size_messi

# Create a list to hold both the original and compressed images
images = [image_ronaldo, reconstructed_ronaldo, image_messi, reconstructed_messi]
titles = ['Original', 'Decompressed', 'Original', 'Decompressed']

# Create a 2x2 grid of subplots for displaying the images
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Iterate through subplots for displaying the images
for i in range(4):
    row = i // 2  # Determine the row
    col = i % 2   # Determine the column
    axes[row, col].imshow(images[i], cmap='gray', norm=Normalize(0, 255))
    axes[row, col].set_title(titles[i])
    axes[row, col].set_xticks([])
    axes[row, col].set_yticks([])

# Define a list of 16 text lines
text_lines = [
    (0.44, 0.85, 'Original size:'),
    (0.44, 0.82, f"{original_image_size_ronaldo} bytes"),
    (0.44, 0.77, 'Encoded size:'),
    (0.44, 0.74, f"{encoded_data_size_ronaldo} bytes"),
    (0.44, 0.69, 'Decoded size:'),
    (0.44, 0.66, f"{decoded_data_size_ronaldo} bytes"),
    (0.44, 0.61, "Compression ratio:"),
    (0.44, 0.58, f"{compression_ratio_ronaldo:.2f}"),
    (0.44, 0.42, 'Original size:'),
    (0.44, 0.39, f"{original_image_size_messi} bytes"),
    (0.44, 0.34, 'Encoded size:'),
    (0.44, 0.31, f"{encoded_data_size_messi} bytes"),
    (0.44, 0.26, 'Decoded size:'),
    (0.44, 0.23, f"{decoded_data_size_messi} bytes"),
    (0.44, 0.19, "Compression ratio:"),
    (0.44, 0.16, f"{compression_ratio_messi:.2f}"),
]

# Add text lines to the entire plot
for x, y, text in text_lines:
    fig.text(x, y, text, fontsize=12, color='black')

# Adjust the horizontal and vertical spacing between subplots
plt.subplots_adjust(wspace=0.2, hspace=0.3)

# Display the subplots
plt.show()