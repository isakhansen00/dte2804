import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

"""
--- Oppgave 5 - SIFT + Morphological operations "Harnverhalt" ---
Denne oppgaven gikk ut på å feature matche ved hjelp av SIFT og å i tillegg kunne gjøre morphologiske
operasjoner på bildeserien vi har fått utlevert. dette for å tydeligere se etter features på bildet.
Programmet fungerer slik:
Man starter programmet og blir møtt med en "meny" hvor man kan trykke tast 1, 2 eller 3.
om man trykker "1" får man mulighet til å utføre "closing" på bildet, dvs. dilation og erotion etter hverandre.
dette gjennomføres på samtlige bilder i bildeserien, og man får et gjennomsnittsresultat etter loopen er kjørt

om man trykker "2" får man en SIFT-analysert utgave av bildeserien.

om man trykker "3" får man en kombinasjon av det morphologiske bildet og den sift analyserte utgaven som overlapper hverandre.

--- REFLEKSJON ---
I etterkant ville det være bedre med færre keypoints i SIFT analysen, da så mange punkter gjøre det betydelig vanskeligere å se
hva som faktisk er relevant og ikke.
"""

image_dir = os.path.abspath(os.path.dirname(__file__))  # Assuming the script is in the same directory as the images

# Function to detect ROIs using morphological operations
def detect_rois_morphological(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply morphological operations (thresholding, erosion, and dilation)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

# Function to detect ROIs using SIFT
def detect_rois_sift(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    
    return keypoints

# Function to combine morphological and SIFT results
def combine_morphological_and_sift(image, morphological_result, keypoints):
    # Convert morphological result to BGR format
    morphological_result_bgr = cv2.cvtColor(morphological_result, cv2.COLOR_GRAY2BGR)
    
    # Overlay keypoints on the morphological result
    result = cv2.drawKeypoints(morphological_result_bgr, keypoints, image)
    
    return result

# Load the image series (harnverhalt1.jpg to harnverhalt14.jpg)

image_series = [cv2.imread(os.path.join(image_dir, f'harnverhalt{i}.png'), cv2.IMREAD_COLOR) for i in range(1, 15)]

# Create a menu to choose the operation
print("Select the type of operation:")
print("1. Morphological Operations")
print("2. SIFT Feature Detection")
print("3. Combination of Morphological and SIFT")

choice = int(input("Enter the operation number (1/2/3): "))

# Process the entire image series based on the chosen operation
if choice == 1:
    results = [detect_rois_morphological(image) for image in image_series]
    result_name = 'Morphological'
elif choice == 2:
    keypoints_list = []
    for image in image_series:
        keypoints = detect_rois_sift(image)
        keypoints_list.append(keypoints)
    
    # Combine all keypoints onto the first image
    combined_image = image_series[0].copy()
    for i in range(1, len(keypoints_list)):
        combined_image = cv2.drawKeypoints(combined_image, keypoints_list[i], combined_image)
    
    result_name = 'SIFT'
elif choice == 3:
    results = []
    for image in image_series:
        morphological_result = detect_rois_morphological(image)
        keypoints = detect_rois_sift(image)
        combined_result = combine_morphological_and_sift(image, morphological_result, keypoints)
        results.append(combined_result)
    result_name = 'Combination of Morphological and SIFT'

# Display the results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

original_image_series = [cv2.imread(os.path.join(image_dir, f'harnverhalt{i}.png'), cv2.IMREAD_COLOR) for i in range(1, 15)]
# Show the first image (original) on the left
ax[0].imshow(original_image_series[0][:, :, ::-1])
ax[0].set_title('Original Image')
ax[0].axis('off')

if choice == 2:
    # Display the combined SIFT result on the right
    ax[1].imshow(combined_image[:, :, ::-1])
    ax[1].set_title(f'Combined {result_name} SIFT Results')
elif choice == 3:
    # Display the combined result for Morphological and SIFT on the right
    ax[1].imshow(results[0][:, :, ::-1])
    ax[1].set_title(f'{result_name} Result for the First Image')
else:
    # Display the processed (average) image on the right
    average_result = np.mean(results, axis=0).astype(np.uint8)
    ax[1].imshow(average_result, cmap='gray' if result_name == 'Morphological' else None)
    ax[1].set_title(f'Average {result_name} Result')

ax[1].axis('off')

plt.tight_layout()
plt.show()