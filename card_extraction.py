import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

BKG_THRESH = 90
PLAYING_CARD_RATIO = 0.72

CORNER_WIDTH = 32
CORNER_HEIGHT = 84

class CardExtractor:

    def __init__(self, img):
        self.img = img

    def get_thresh_img(self):


        img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(img, (5, 5), 0)

        use_background = True
        if use_background:
            img_w, img_h = np.shape(img)[:2]
            bkg_level = img[int(img_h / 100)][int(img_w / 2)]
            thresh_level = bkg_level + BKG_THRESH

            retval, thresh = cv.threshold(blur, thresh_level, 255, cv.THRESH_BINARY)
        else:
            _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        return thresh

    def is_rectangle(self, cnt):
        size = cv.contourArea(cnt)
        rect = cv.minAreaRect(cnt)
        w, h = rect[1][0], rect[1][1]
        if not np.isclose(w * h, size, rtol=0.2):
            return False

        return True

    def flatten(self, pts, w, h):
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
        warp = cv.warpPerspective(self.img, M, (maxWidth, maxHeight))
        warp = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)

        return warp

    def get_cards(self):
        thresh_img = self.get_thresh_img()

        im2, contours, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_im = cv.cvtColor(np.copy(thresh_img), cv.COLOR_GRAY2RGB)
        contour_im = cv.drawContours(contour_im, contours, -1, (0, 255, 0), 20)
        plt.imshow(contour_im)
        plt.xticks([]), plt.yticks([])
        plt.show()

        warped_cards = []
        for cnt in contours:
            eps = 0.01 * cv.arcLength(cnt, True)
            pts = cv.approxPolyDP(cnt, eps, True)
            if not self.is_rectangle(cnt):
                continue

            x, y, w, h = cv.boundingRect(cnt)

            warped_cards.append(self.flatten(pts, w, h))

        return warped_cards


class CardAnalysis:
    def __init__(self, card_img):
        self.card_img = card_img

        self.corner_img = card_img[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
        rsz_corner = np.array(cv.resize(self.corner_img, (0, 0), fx=4, fy=4))
        corner_blur = cv.GaussianBlur(rsz_corner, (5, 5), 0)
        ret3, self.corner_img = cv.threshold(corner_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        plt.imshow(self.corner_img, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

        _, corner_cnt, _ = cv.findContours(self.corner_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.corner_cnts, self.bounding_boxes, areas = self.get_corner_contours(corner_cnt)

    def get_corner_contours(self, cnts):
        boundingBoxes = [cv.boundingRect(c) for c in cnts]
        areas = [cv.contourArea(c) for c in cnts]
        combined = zip(cnts, boundingBoxes, areas)
        area_filter = list(filter(lambda x: x[2] > 1000 and x[1][0] < 60, combined))
        objects = sorted(area_filter, key=lambda b: b[1][1], reverse=False)
        if len(objects) == 3:
            # is a ten, merge the first 1 and 0 to make 10
            merged_cnts = cv.convexHull(np.concatenate((objects[0][0], objects[1][0])))
            merged_rect = cv.boundingRect(merged_cnts)
            merged_area = cv.contourArea(merged_cnts)
            return [merged_cnts, objects[2][0]], [merged_rect, objects[2][1]], [merged_area, objects[2][2]]
        return zip(*objects)


    def get_rank_img(self):
        x, y, w, h = self.bounding_boxes[0]
        rank_img = self.corner_img[y:y + h, x:x + w]
        rank_img = cv.resize(rank_img, (80, 130), 0, 0)
        return rank_img

    def get_suit_img(self):
        x, y, w, h = self.bounding_boxes[1]
        suit_img = self.corner_img[y:y + h, x:x + w]
        suit_img = cv.resize(suit_img, (100, 130), 0, 0)
        return suit_img

