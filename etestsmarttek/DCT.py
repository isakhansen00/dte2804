import cv2
import numpy as np

import cv2
import numpy as np

def image_to_dct():
    # Read the image in grayscale mode using OpenCV
    img = cv2.imread("sui.png", cv2.IMREAD_GRAYSCALE)
    ''''''
    img = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73],
    [63, 59, 55, 90, 109, 85, 69, 72],
    [62, 59, 68, 213, 144, 104, 66, 73],
    [63, 58, 71, 222, 154, 106, 70, 69],
    [67, 61, 68, 104, 126, 88, 68, 70],
    [79, 65, 160, 170, 177, 168, 58, 75],
    [85, 71, 164, 159, 155, 161, 65, 83],
    [87, 79, 169, 168, 165, 176, 78, 94]
    ], dtype=np.float32)
    # Resize the image to 8x8
    #img = cv2.resize(img, (8, 8))
    # Perform DCT on the resized image
    # Perform DCT on the resized image
    dct_matrix = cv2.dct(np.float32(img))
    
    # Normalize DCT coefficients to be in the range [0, 255]
    
    return dct_matrix

def dequantize_dct(dct_matrix, quantization_matrix):
    # Reverse quantization by multiplying with the quantization matrix
    decompressed_dct_matrix = np.multiply(dct_matrix, quantization_matrix)
    
    return decompressed_dct_matrix

def dct_to_image(dct_matrix):
    # Perform inverse DCT to obtain the image
    img = cv2.idct(np.float32(dct_matrix))
    
    # Clip the values to be within the valid pixel range [0, 255]
    img = np.clip(img, 0, 255)

    # Convert back to uint8 data type
    img = np.uint8(img)
    return img

 #Define the standard JPEG quantization matrix
quantization_matrix = np.array([
     [100, 100, 100, 90, 80, 20, 20, 20],
     [100, 100, 100, 100, 100, 20, 100, 2],
     [100, 100, 100, 90, 80, 80, 20, 40],
     [100, 100, 100, 100, 80, 100, 100, 40],
     [100, 100, 100, 100, 100, 100, 20, 10],
     [70, 80, 100, 100, 100, 20, 2, 10],
     [20, 100, 100, 100, 100, 100, 3, 2],
     [2, 100, 100, 90, 70, 20, 2, 2]
 ])
'''
quantization_matrix = np.array([
     [3, 5, 7, 9, 11, 13, 15, 17],
     [5, 7, 9, 11, 13, 15, 17, 19],
     [7, 9, 11, 13, 15, 17, 19, 21],
     [9, 11, 13, 15, 17, 19, 21, 23],
     [11, 13, 15, 17, 19, 21, 23, 25],
     [13, 15, 17, 19, 21, 23, 25, 27],
     [15, 17, 19, 21, 23, 25, 27, 29],
     [17, 19, 21, 23, 25, 27, 29, 31]
 ])

'''
# Example usage
normalized_dct_coefficient_matrix = image_to_dct()

quantized_dct_matrix = np.round(normalized_dct_coefficient_matrix / quantization_matrix)
print(quantized_dct_matrix)
# Dequantize the DCT matrix
decompressed_dct_matrix = dequantize_dct(quantization_matrix, quantized_dct_matrix)
print(normalized_dct_coefficient_matrix)
print(decompressed_dct_matrix)

# Perform inverse DCT using OpenCV to obtain the image
reconstructed_image = cv2.idct(np.float32(decompressed_dct_matrix))
print(reconstructed_image)

# Convert to uint8 data type
reconstructed_image = np.uint8(reconstructed_image)
resized_image = cv2.resize(reconstructed_image, (129, 129))

img = cv2.imread("sui.png", cv2.IMREAD_GRAYSCALE)

img = np.array([
[52, 55, 61, 66, 70, 61, 64, 73],
[63, 59, 55, 90, 109, 85, 69, 72],
[62, 59, 68, 213, 144, 104, 66, 73],
[63, 58, 71, 222, 154, 106, 70, 69],
[67, 61, 68, 104, 126, 88, 68, 70],
[79, 65, 160, 170, 177, 168, 58, 75],
[85, 71, 164, 159, 155, 161, 65, 83],
[87, 79, 169, 168, 165, 176, 78, 94]
], dtype=np.float32)
    # Clip the values to be within the valid pixel range [0, 255]
img = np.clip(img, 0, 255)

    # Convert back to uint8 data type
img = np.uint8(img)
img = cv2.resize(img, (129, 129))
cv2.imshow("Original", img)

# Display or save the reconstructed image
cv2.imshow('Reconstructed Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()