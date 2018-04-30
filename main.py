import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

BKG_THRESH = 90

img = cv.imread('2cards.jpg', 0)
blur = cv.GaussianBlur(img, (5,5),0)
img_w, img_h = np.shape(img)[:2]
bkg_level = img[int(img_h/100)][int(img_w/2)]
thresh_level = bkg_level + BKG_THRESH

retval, thresh = cv.threshold(blur,thresh_level,255,cv.THRESH_BINARY)
edges = cv.Canny(thresh, 50, 220)

surf = cv.xfeatures2d.SURF_create(4000)
kp, des = surf.detectAndCompute(edges, None)

#gray = np.float32(edges)
#dst = cv.cornerHarris(gray, 2, 3, 0.04)
#result is dilated for marking the corners, not important
#dst = cv.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
#img[dst>0.2*dst.max()]=255

img2 = cv.drawKeypoints(img, kp, None, (255,0,0), 4)

plt.imshow(img2, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.show()



