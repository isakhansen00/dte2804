import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Load the two DICOM images
image1 = pydicom.dcmread('etestsmarttek/0173.dcm')
image2 = pydicom.dcmread('etestsmarttek/0174.dcm')

# Extract pixel arrays from the DICOM images
pixel_array1 = image1.pixel_array
pixel_array2 = image2.pixel_array

# Ensure both images have the same dimensions
if pixel_array1.shape != pixel_array2.shape:
    raise ValueError("Images have different dimensions.")

# Calculate the absolute difference between pixel arrays
pixel_diff = np.abs(pixel_array1 * 0.5 - pixel_array2 * 0.5)

# Convert the pixel difference to an 8-bit image (clipping values)
pixel_diff_8bit = np.clip(pixel_diff, 0, 255).astype(np.uint8)

# Create a list to hold both the frame images and pixel difference image
images = []

images.append(pixel_array1)
images.append(pixel_array2)
images.append(pixel_diff_8bit)

titles = ['Frame 1', "Frame 2", "Pixel Difference"]

# Create a 1x3 grid of subplots for displaying the images and text
fig = plt.figure(figsize=(15, 7.5))

# Define the vertical position for the images
image_position = 0.5  # Adjust this value to move images closer to the top

# Define the size and position of each subplot
image_width = 0.4  # Adjust this value to make images wider
image_height = 0.4  # Adjust this value to make images taller

subplot1 = fig.add_axes([0.0, image_position, image_width, image_height])
subplot2 = fig.add_axes([0.3, image_position, image_width, image_height])
subplot3 = fig.add_axes([0.6, image_position, image_width, image_height])

# Display the images
subplot1.imshow(images[0], 'gray')
subplot2.imshow(images[1], 'gray')
subplot3.imshow(images[2], 'gray')

# Set titles for the subplots
subplot1.set_title(titles[0])
subplot2.set_title(titles[1])
subplot3.set_title(titles[2])

# Remove axis labels
subplot1.set_xticks([])
subplot1.set_yticks([])
subplot2.set_xticks([])
subplot2.set_yticks([])
subplot3.set_xticks([])
subplot3.set_yticks([])

# Add text below the images
text = "Er dette en P-frame?"
subplot1.text(0.0, -0.50, text, transform=subplot1.transAxes, ha='left', va='bottom', fontsize=10, weight='bold', wrap=True)
text2 = "Nei, pixel difference-bildet som er laget fra pikselforskjellene mellom to DICOM-bilder er ikke en P-ramme. En P-ramme (Predictive Frame) er en spesifikk type ramme som brukes i videokomprimering, spesielt i videokodingsstandarder som H.264 (MPEG-4 AVC) eller H.265 (HEVC). P-rammer brukes vanligvis i videokomprimering for å representere forskjellene eller endringene mellom en referanseramme (vanligvis en I-ramme eller en tidligere P-ramme) og gjeldende ramme, i stedet for å lagre hver piksel individuelt. Pixel difference-bildet er en representasjon av de absolutte pikselforskjellene mellom to individuelle DICOM-bilder. Den har ikke de tidsmessige prediktive egenskapene som finnes i P-rammer til videokodeker. Det er egentlig et statisk bilde som viser størrelsen på pikselforskjellene mellom de to inngangsbildene, og det er ikke en del av et videokomprimeringsskjema."
subplot1.text(0.0, -0.81, text2, transform=subplot1.transAxes, ha='left', va='bottom', fontsize=10, wrap=True)
text3 = "Hvorfor er det så viktig å se pikselverdiforskjeller mellom påfølgende bilder i den virkelige verden?"
subplot1.text(0.0, -0.15, text3, transform=subplot1.transAxes, ha='left', va='bottom', fontsize=10, weight='bold', wrap=True)
text4 = "Differensialrammer, ofte referert til som bevegelsesdeteksjon eller rammeforskjell, er mye brukt i forskjellige virkelige applikasjoner. De er spesielt nyttige i scenarier der du ønsker å oppdage endringer mellom påfølgende rammer, for eksempel i systemer for sikkerhet, overvåking og objektsporing. På bildet over blir det brukt til å sjekke om en svulst har vokst i hjernen, noe som vil hjelpe til med å oppdage den tidligere selv om den er i et tidlig stadie"
subplot1.text(0.0, -0.35, text4, transform=subplot1.transAxes, ha='left', va='bottom', fontsize=10, wrap=True)

# Adjust the horizontal spacing between subplots
plt.subplots_adjust(wspace=0.1)

# Display the subplots
plt.show()