import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import math

BKG_THRESH = 90
PLAYING_CARD_RATIO = 0.72

CORNER_WIDTH = 32
CORNER_HEIGHT = 84


def is_rectangle(cnt, pts):
  size = cv.contourArea(cnt)
  rect = cv.minAreaRect(cnt)
  w, h = rect[1][0], rect[1][1]
  if not np.isclose(w*h, size, rtol=0.2):
    return False

  return True


def flattener(image, pts, w, h):
  """Flattens an image of a card into a top-down 200x300 perspective.
  Returns the flattened, re-sized, grayed image.
  See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
  temp_rect = np.zeros((4, 2), dtype="float32")

  s = np.sum(pts, axis=2)

  tl = pts[np.argmin(s)]
  br = pts[np.argmax(s)]

  diff = np.diff(pts, axis=-1)
  tr = pts[np.argmin(diff)]
  bl = pts[np.argmax(diff)]

  # Need to create an array listing points in order of
  # [top left, top right, bottom right, bottom left]
  # before doing the perspective transform

  if w <= 0.8 * h:  # If card is vertically oriented
    temp_rect[0] = tl
    temp_rect[1] = tr
    temp_rect[2] = br
    temp_rect[3] = bl

  if w >= 1.2 * h:  # If card is horizontally oriented
    temp_rect[0] = bl
    temp_rect[1] = tl
    temp_rect[2] = tr
    temp_rect[3] = br

  # If the card is 'diamond' oriented, a different algorithm
  # has to be used to identify which point is top left, top right
  # bottom left, and bottom right.

  if w > 0.8 * h and w < 1.2 * h:  # If card is diamond oriented
    # If furthest left point is higher than furthest right point,
    # card is tilted to the left.
    if pts[1][0][1] <= pts[3][0][1]:
      # If card is titled to the left, approxPolyDP returns points
      # in this order: top right, top left, bottom left, bottom right
      temp_rect[0] = pts[1][0]  # Top left
      temp_rect[1] = pts[0][0]  # Top right
      temp_rect[2] = pts[3][0]  # Bottom right
      temp_rect[3] = pts[2][0]  # Bottom left

    # If furthest left point is lower than furthest right point,
    # card is tilted to the right
    if pts[1][0][1] > pts[3][0][1]:
      # If card is titled to the right, approxPolyDP returns points
      # in this order: top left, bottom left, bottom right, top right
      temp_rect[0] = pts[0][0]  # Top left
      temp_rect[1] = pts[3][0]  # Top right
      temp_rect[2] = pts[2][0]  # Bottom right
      temp_rect[3] = pts[1][0]  # Bottom left

  maxWidth = 200
  maxHeight = 300

  # Create destination array, calculate perspective transform matrix,
  # and warp card image
  dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
  M = cv.getPerspectiveTransform(temp_rect, dst)
  warp = cv.warpPerspective(image, M, (maxWidth, maxHeight))
  warp = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)

  return warp

img_orig = cv.imread('2cards.jpg')
img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(img, (5,5),0)
img_w, img_h = np.shape(img)[:2]
bkg_level = img[int(img_h/100)][int(img_w/2)]
thresh_level = bkg_level + BKG_THRESH

retval, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contour_im = np.copy(img_orig)
warped = []
for cnt in contours:
  eps = 0.01 * cv.arcLength(cnt, True)
  pts = cv.approxPolyDP(cnt, eps, True)
  if not is_rectangle(cnt, pts):
    continue

  x,y,w,h = cv.boundingRect(cnt)
  contour_im = cv.drawContours(contour_im, [pts], -1, (0, 255, 0), 3)
  warped.append(flattener(img_orig, pts, w, h))

show_all = False
if show_all:
  plt.imshow(contour_im, cmap='gray')
  cv.imwrite('contour_map.jpg', contour_im)
  plt.xticks([]), plt.yticks([])
  plt.show()

cv.imwrite('card1.jpg', warped[0])
cv.imwrite('card2.jpg', warped[1])

show_all_extracted = False
if show_all_extracted:
  n = len(warped)
  for i, warp in enumerate(warped):
    plt.subplot(math.ceil(n/12), 12, i+1)
    plt.imshow(warp, cmap='gray')
    plt.xticks([]), plt.yticks([])

  plt.show()


warp = warped[0]
corner = warped[0][0:CORNER_HEIGHT, 0:CORNER_WIDTH]
rsz_corner = np.array(cv.resize(corner, (0,0), fx=4, fy=4))
corner_blur = cv.GaussianBlur(rsz_corner, (5,5), 0)
ret3, corner_thresh = cv.threshold(corner_blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)


top_half = corner_thresh[25:CORNER_HEIGHT*2+10, 0:128]
bottom_half = corner_thresh[CORNER_HEIGHT*2+11:CORNER_HEIGHT*4-50, 0:128]

show_one_extracted = False
if show_one_extracted:
  plt.imshow(corner_thresh, cmap='gray')
  plt.xticks([]), plt.yticks([])
  plt.show()


_, top_cnt, _ = cv.findContours(top_half, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
_, bot_cnt, _ = cv.findContours(bottom_half, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

top_cnt = sorted(top_cnt, key=cv.contourArea, reverse=True)
bot_cnt = sorted(bot_cnt, key=cv.contourArea, reverse=True)

if len(top_cnt) > 0: # Rank
  largest_cnt = top_cnt[0]
  x,y,w,h = cv.boundingRect(largest_cnt)
  rank_img = top_half[y:y+h, x:x+w]
  rank_img = cv.resize(rank_img, (100, 130), 0, 0)

  plt.imshow(rank_img, cmap='gray')
  plt.xticks([]), plt.yticks([])
  plt.show()

if len(bot_cnt) > 0: # Suit
  largest_cnt = bot_cnt[0]
  x, y, w, h = cv.boundingRect(largest_cnt)
  rank_img = bottom_half[y:y + h, x:x + w]
  rank_img = cv.resize(rank_img, (100, 115), 0, 0)

  #plt.imshow(rank_img, cmap='gray')
  #plt.xticks([]), plt.yticks([])
  #plt.show()


