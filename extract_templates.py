import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import math
from card_extraction import CardExtractor, CardAnalysis

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['S', 'C', 'D', 'H']
img_orig = cv.imread('templates_imgs/9diamonds.jpg')

pct = 0.2
newsize = (int(img_orig.shape[1] * pct), int(img_orig.shape[0] * pct))
img_orig = cv.resize(img_orig, newsize)

cards = CardExtractor(img_orig).get_cards()

first_card = cards[0]
card_analysis = CardAnalysis(first_card)

rank_img = card_analysis.get_rank_img()
plt.imshow(rank_img, cmap='gray')
cv.imwrite('templates/9.jpg', rank_img)
plt.xticks([]), plt.yticks([])
plt.show()

"""
suit_img = card_analysis.get_suit_img()
plt.imshow(suit_img, cmap='gray')
cv.imwrite('templates/s.jpg', suit_img)
plt.xticks([]), plt.yticks([])
plt.show()
"""
