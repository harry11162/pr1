'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import os, cv2
from roipoly import RoiPoly
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle
import numpy as np
from skimage import transform
from bin_detector import BinDetector

if __name__ == '__main__':

  image_folder = 'data/training'
  mask_folder = 'mask'

  contours_to_use = []

  for i in range(1, 51):

    img = cv2.imread(os.path.join(image_folder, '00{:02d}.jpg'.format(i)))
    h, w, _ = img.shape
    # balance the image with mean brightness
    # brightness = img * np.array([0.2126, 0.7152, 0.0722])[np.newaxis, np.newaxis, :]
    # brightness = brightness.sum(axis=2).mean()
    # img = img / (brightness * 1e-2)

    with open(os.path.join(mask_folder, "mask_00{:02d}.pkl".format(i)), "rb") as f:
      mask = pickle.load(f)
    img = img * mask[:, :, np.newaxis]

    detector = BinDetector()
    mask = detector.segment_image(img)

    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_FLOODFILL, method=cv2.CHAIN_APPROX_TC89_L1)

    max_contour_area = -1.
    sum_area = 0
    for c in contours:
      area = cv2.contourArea(c)
      sum_area += area
      if area > max_contour_area:
        max_contour_area = area
        max_contour = c
    
    if max_contour_area / sum_area > 0.4:
      contours_to_use.append(max_contour)

with open("known_contours.pkl", "wb") as f:
  pickle.dump(contours_to_use, f)