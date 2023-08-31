import cv2 as cv
import sys
from matplotlib import pyplot as plt
img = cv.imread("openCVPython/image/cat2.jpg", cv.IMREAD_GRAYSCALE)

if img is None:
    sys.exit("Could not read image")

# cv.imshow("Katt", img)
# k = cv.waitKey(0)

edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# if k == ord("s"):
#     cv.imwrite("images/starry_night.jpg")