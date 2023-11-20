# import numpy as np
# import matplotlib.pyplot as plt
# import pywt
# import pywt.data

# import cv2

# # Load an example image
# image_path = "sui.png"
# n=2
# original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Flatten the 2D image into a 1D array
# original_signal = original_image.flatten()

# # 1D Discrete Wavelet Transform (DWT)
# coeffs = pywt.dwt(original_signal, 'db1')

# # Choose a specific level to embed meta information
# meta_level = 1

# # Get the coefficients at the chosen level
# coeffs_list = list(coeffs)

# # Get the shape of the coefficients at the chosen level
# coeffs_shape = coeffs_list[meta_level].shape

# # Generate a binary signature (you can replace this with your own signature creation logic)
# signature = np.random.randint(0, 25, coeffs_shape)

# # Embed signature in the wavelet coefficients
# coeffs_list[meta_level] = coeffs_list[meta_level] + signature

# # Convert back to tuple
# coeffs = tuple(coeffs_list)

# # Inverse 1D Discrete Wavelet Transform (IDWT)
# modified_signal = pywt.idwt(coeffs[0], coeffs[meta_level], 'db1')

# # Reshape the modified signal back to the 2D image shape
# modified_image = modified_signal.reshape(original_image.shape)

# # Plot original and modified images
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(original_image, cmap='gray')
# plt.title('Original Image')

# plt.subplot(1, 2, 2)
# plt.imshow(modified_image.astype('uint8'), cmap='gray')
# plt.title('Modified Image with Signature')
# plt.show()


import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

def embed_meta_info(image, meta_info):
    # Perform DWT on the image
    coeffs = pywt.dwt2(image, 'haar')

    # Embed meta information in the LL subband
    LL_subband = coeffs[0]
    LL_subband[:meta_info.shape[0], :meta_info.shape[1]] = meta_info

    # Reconstruct the image
    reconstructed_image = pywt.idwt2(coeffs, 'haar')

    return reconstructed_image

def main():
    image_path = "sui.png"
    n=2
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create a small square as meta information
    meta_info_size = 100
    meta_info = np.ones((meta_info_size, meta_info_size), dtype=np.uint8)
    print(meta_info)

    # Embed meta information
    new_image = embed_meta_info(original_image, meta_info)

    # Plot the original and the new image
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(new_image, cmap='gray')
    plt.title('Image with Embedded Meta Info')

    plt.show()

if __name__ == "__main__":
    main()
