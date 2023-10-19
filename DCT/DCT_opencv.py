import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# Function to calculate the Mean Absolute Difference (MAD) between two images
def calculate_mad(image1, image2):
    return np.mean(np.abs(image1 - image2))


# Load the image
image = cv2.imread('lab/Part A_DCT_Fourier_wavelet/DCT/minions.jpg', cv2.IMREAD_GRAYSCALE)

# Define the block size
B = 8

# Crop the image to ensure its dimensions are multiples of B
height, width = image.shape
new_height = (height // B) * B
new_width = (width // B) * B
cropped_image = image[:new_height, :new_width]

# Initialize variables to store the transformed and reconstructed images
Trans = np.zeros(cropped_image.shape, dtype=np.float32)
back0 = np.zeros(cropped_image.shape, dtype=np.float32)

# Perform the DCT and IDCT block by block
for y in range(0, new_height, B):
    for x in range(0, new_width, B):
        block = cropped_image[y:y + B, x:x + B]

        # Apply DCT
        dct_block = cv2.dct(np.float32(block))
        Trans[y:y + B, x:x + B] = dct_block

        # Apply IDCT
        idct_block = cv2.idct(dct_block)
        back0[y:y + B, x:x + B] = idct_block

# Save the transformed image
cv2.imwrite('Transformed.jpg', Trans)

# Create a Matplotlib figure to display the original image
plt.figure(figsize=(8, 6))
plt.imshow(cropped_image, cmap='gray', norm=Normalize(0, 255))
plt.title('Original Image')


# Function to handle mouse click event for block selection

def onclick(event):
    global selected_block_coords

    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        selected_block_coords = (x, y)

        # Redraw the original image with the selected block highlighted in red
        plt.figure(figsize=(8, 6))
        plt.imshow(cropped_image, cmap='gray', norm=Normalize(0, 255))
        plt.title('Original Image')

        # Highlight the selected block with a red frame
        if selected_block_coords is not None:
            x, y = selected_block_coords
            plt.gca().add_patch(plt.Rectangle((x, y), B, B, fill=False, edgecolor='red', linewidth=2))

        # Display the DCT of the selected block
        if selected_block_coords is not None:
            x, y = selected_block_coords
            selected_block = cropped_image[y:y + B, x:x + B]
            selected_dct_block = Trans[y:y + B, x:x + B]

            # Create a Matplotlib figure to display the selected block and its DCT
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(selected_block, cmap='gray', norm=Normalize(0, 255))
            plt.title('Selected Block')
            plt.subplot(122)
            plt.imshow(selected_dct_block, cmap=cm.jet, norm=Normalize(vmin=np.min(selected_dct_block), vmax=np.max(selected_dct_block)))
            plt.title('DCT of Selected Block')
            plt.colorbar()

        plt.show()

# Connect the mouse click event to the function
plt.connect('button_press_event', onclick)

# Display the original image
plt.show()

# Save the reconstructed image
cv2.imwrite('BackTransformed.jpg', back0)

# Calculate the Mean Absolute Difference (MAD) between the original and reconstructed images
mad_value = calculate_mad(cropped_image, back0)
print(f'Mean Absolute Difference (MAD) between original and reconstructed images: {mad_value}')

# Create a Matplotlib figure to display the reconstructed image
plt.figure(figsize=(8, 6))
plt.imshow(back0, cmap='gray', norm=Normalize(0, 255))
plt.title('Reconstructed Image')
plt.show()
