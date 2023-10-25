import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import os

# Get the directory where the script is located
image_dir = os.path.abspath(os.path.dirname(__file__))

# Create a GUI window
root = Tk()
root.title("ROI Detection")

# Create a label
label = Label(root, text="Select Morphological Operation:")
label.pack()

# Create a dropdown menu for morphological operations
operations = ["None", "Closing", "Erosion", "Dilation"]
operation_var = StringVar()
operation_var.set(operations[0])  # Set the default option to "None"
operation_menu = OptionMenu(root, operation_var, *operations)
operation_menu.pack()

# Create radio buttons for SIFT
sift_var = IntVar()
sift_checkbox = Checkbutton(root, text="Use SIFT", variable=sift_var)
sift_checkbox.pack()

# Function to load images and perform detection
def detect_roi():
    # Construct file paths for the template and image series using os.path.join and image_dir
    template_path = os.path.join(image_dir, "matTeil.jpg")
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    # Get the selected morphological operation
    morph_operation = operation_var.get()

    # Set a threshold for template matching
    threshold = 0.65

    # Create a grid of plots
    num_columns = 4  # Number of columns in the grid
    num_rows = 4     # Number of rows in the grid

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 15))
    fig.suptitle("ROI Detection Results")

    for i in range(1, 15):

        # Construct the file path for the current image using os.path.join and image_dir
        image_filename = f"harnverhalt{i}.png"
        image_path = os.path.join(image_dir, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply the selected morphological operation
        if morph_operation == "Closing":
            kernel = np.ones((5, 5), np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif morph_operation == "Erosion":
            kernel = np.ones((5, 5), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)
        elif morph_operation == "Dilation":
            kernel = np.ones((5, 5), np.uint8)
            image = cv2.dilate(image, kernel, iterations=1)
        elif morph_operation == "None":
            pass  # No morphological operation

        # If SIFT is selected, perform SIFT matching
        if sift_var.get() == 1:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(template, None)
            kp2, des2 = sift.detectAndCompute(image, None)

            # FLANN parameters
            flann_index_kdtree = 0
            index_params = dict(algorithm=flann_index_kdtree, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Draw matched lines
            img_matches = cv2.drawMatches(template, kp1, image, kp2, good_matches, None)
            row, col = (i - 1) // num_columns, (i - 1) % num_columns
            axes[row, col].imshow(img_matches)
            axes[row, col].set_title(f"Similarity Lines - {image_filename}", fontsize=10)

        # If SIFT is not selected, use template matching
        else:
            res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color image

            for pt in zip(*loc[::-1]):
                cv2.rectangle(image_color, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)

            row, col = (i - 1) // num_columns, (i - 1) % num_columns
            axes[row, col].imshow(image_color)
            axes[row, col].set_title(f"ROI Detection for {image_filename}", fontsize=10)

    # Hide empty subplots
    for i in range(14, num_columns * num_rows):
        row, col = i // num_columns, i % num_columns
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Create a button to start detection
detect_button = Button(root, text="Detect ROI", command=detect_roi)
detect_button.pack()

# Start the GUI loop
root.mainloop()
