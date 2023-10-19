import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Load the two DCM images
image1 = pydicom.dcmread('etestsmarttek/0173.dcm').pixel_array
image2 = pydicom.dcmread('etestsmarttek/0174.dcm').pixel_array

# Calculate pixel value differences
differential_frame = np.abs(image2*0.6 - image1*0.6)


plt.show()

fig = plt.figure(figsize=(15, 7.5))


image_position = 0.5  # Adjust this value to move images closer to the top


image_width = 0.4 
image_height = 0.4 

subplot1 = fig.add_axes([0.3, image_position, image_width, image_height])

subplot1.imshow(differential_frame, 'gray')
subplot1.set_title("Pixel difference image")
subplot1.set_xticks([])
subplot1.set_yticks([])

text = "Hvorfor er denne teknikken så viktig?"
subplot1.text(-1.5, -0.50, text, transform=subplot1.transAxes, ha='left', va='bottom', fontsize=10, weight='bold', wrap=True)
text2 = "differential frames er viktig i flere bruksområder, for eksempel i medisink avbildning hvor man analyserer ct og mr skanninger, ved å kalkulere differential frames for å se etter endringer i for eksempel vev. Kvalietskontroll er et annet bruksområde hvor differential frames kan brukes for å detektere defekter i produktet i forhold til hva som er rett."
subplot1.text(-1.5, -0.81, text2, transform=subplot1.transAxes, ha='left', va='bottom', fontsize=10, wrap=True)
text3 = "Er dette en P-frame"
subplot1.text(-1.5, -0.15, text3, transform=subplot1.transAxes, ha='left', va='bottom', fontsize=10, weight='bold', wrap=True)
text4 = "Nei, dette er ikke en P-frame (predictive frame), en P-frame lages ved å predikere neste bilde basert på et tidligere bilde. I vårt tilfelle skjer ingen predikering, vi bare tar verdien av et bilde minus et annet bilde for å finne differensen"
subplot1.text(-1.5, -0.35, text4, transform=subplot1.transAxes, ha='left', va='bottom', fontsize=10, wrap=True)

plt.subplots_adjust(wspace=0.1)


plt.show()