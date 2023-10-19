import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
import cv2

n = 4
w = 'db1'
img = cv2.imread('sui.png', cv2.IMREAD_GRAYSCALE)

coeffs = pywt.wavedec2(img, wavelet=w, level=n)

coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

for keep_w in (0.1, 0.05, 0.01, 0.005):
    thresh = Csort[int(np.floor((1-keep_w)*len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind # Threshold small indices

    coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')


    #Plot reconstruction
    Arecon = pywt.waverec2(coeffs_filt, wavelet=w)
    #cv2.imwrite('compressed_sui.jpg', Arecon)
    plt.imshow(Arecon.astype('uint8'), cmap='gray')
    plt.title(keep_w)
    plt.show()