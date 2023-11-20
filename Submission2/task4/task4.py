
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
