import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import math
from card_extraction import CardExtractor, CardAnalysis

img_orig = cv.imread('2hearts.jpg')

pct = 0.2
newsize = (int(img_orig.shape[1] * pct), int(img_orig.shape[0] * pct))
img_orig = cv.resize(img_orig, newsize)

cards = CardExtractor(img_orig).get_cards()

first_card = cards[0]
card_analysis = CardAnalysis(first_card)

plt.imshow(card_analysis.get_rank_img(), cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(card_analysis.get_suit_img(), cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()


show_all = True
if show_all:
  plt.imshow(cards[0], cmap='gray')
  #cv.imwrite('contour_map.jpg', contour_im)
  plt.xticks([]), plt.yticks([])
  plt.show()

show_all_extracted = False
if show_all_extracted:
  n = len(warped)
  for i, warp in enumerate(warped):
    plt.subplot(math.ceil(n/12), 12, i+1)
    plt.imshow(warp, cmap='gray')
    plt.xticks([]), plt.yticks([])

  plt.show()


