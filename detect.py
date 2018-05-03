import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import math
from card_extraction import CardExtractor, CardAnalysis

rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k', 'a']
suit_names = ['s', 'd', 'c', 'h']

ranks = {}
suits = {}

for rank in rank_names:
    i = cv.imread('templates/' + rank + '.jpg', cv.IMREAD_GRAYSCALE)
    ranks[rank] = i

for suit in suit_names:
    i = cv.imread('templates/' + suit + '.jpg', cv.IMREAD_GRAYSCALE)
    suits[suit] = i

def classify_rank(rank_img):
    rank_name = 'U'
    min_rank_score = 99999
    for rank, template_img in ranks.items():
        diff_img = cv.absdiff(rank_img, template_img)
        rank_diff = int(np.sum(diff_img) / 255)

        if rank_diff < min_rank_score:
            rank_name = rank
            min_rank_score = rank_diff

    return rank_name, min_rank_score

def classify_suit(suit_img):
    suit_name = 'U'
    min_suit_score = 99999
    for suit, template_img in suits.items():
        plt.imshow(suit_img, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()
        plt.imshow(template_img, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()
        diff_img = cv.absdiff(suit_img, template_img)
        suit_diff = int(np.sum(diff_img) / 255)

        if suit_diff < min_suit_score:
            suit_name = suit
            min_suit_score = suit_diff

    return suit_name, min_suit_score

img_orig = cv.imread('2_skewed.jpg')

downsample = False
if downsample:
    pct = 0.2
    newsize = (int(img_orig.shape[1] * pct), int(img_orig.shape[0] * pct))
    img_orig = cv.resize(img_orig, newsize)

cards = CardExtractor(img_orig).get_cards()

for card in cards:
    plt.imshow(card, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    card_analysis = CardAnalysis(card)

    rank_img = card_analysis.get_rank_img()
    pred_rank, rank_score = classify_rank(rank_img)

    suit_img = card_analysis.get_suit_img()
    pred_suit, suit_score = classify_suit(suit_img)

    print(pred_rank.upper() + " - " + pred_suit.upper())