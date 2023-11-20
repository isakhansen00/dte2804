import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
import cv2

image_path = "sui.png"
n=2
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


coeffs2 = pywt.dwt2(original_image, 'haar')
LL, (LH, HL, HH) = coeffs2


titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
images = [LL, LH, HL, HH]

fig = plt.figure(figsize=(12, 3))
for i, (image, title) in enumerate(zip(images, titles)):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(np.asarray(image), cmap=plt.cm.gray)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
