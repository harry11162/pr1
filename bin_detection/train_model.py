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

if __name__ == '__main__':

  image_folder = 'data/training'
  mask_folder = 'mask'

  pos_data = np.empty((0, 3))
  neg_data = np.empty((0, 3))

  contours = []

  for i in range(1, 61):

    img = cv2.imread(os.path.join(image_folder, '00{:02d}.jpg'.format(i)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    # img = img.astype(np.float64)/255

    # balance the image with mean brightness
    # brightness = img * np.array([0.2126, 0.7152, 0.0722])[np.newaxis, np.newaxis, :]
    # brightness = brightness.sum(axis=2).mean()
    # img = img / (brightness * 1e-2)

    h, w, c = img.shape

    if i <= 50:
      with open(os.path.join(mask_folder, "mask_00{:02d}.pkl".format(i)), "rb") as f:
        mask = pickle.load(f)
    else:
      mask = np.zeros((h, w), dtype=bool)

    mask = transform.resize(mask, (512, 512)) > 0.5

    img_to_show = img * mask[:, :, np.newaxis]
    img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', img_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if i <= 50:
      contour, hierarchy = cv2.findContours(mask.astype(np.uint8), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_TC89_L1)
      if i == 44:
        contour = [contour[-1]]
      assert len(contour) == 1
      contours = contours + contour
    
    pos_data = np.concatenate((pos_data, img[mask]), axis=0)
    neg_data = np.concatenate((neg_data, img[~mask]), axis=0)

  n_pos = pos_data.shape[0]
  n_neg = neg_data.shape[0]
  theta = (n_pos / (n_pos + n_neg), n_neg / (n_pos + n_neg))
  pos_mean = pos_data.mean(axis=0)
  neg_mean = neg_data.mean(axis=0)
  pos_cov = np.cov(pos_data.T)
  neg_cov = np.cov(neg_data.T)

  print(theta)
  print(pos_mean.tolist())
  print(neg_mean.tolist())
  print(pos_cov.tolist())
  print(neg_cov.tolist())
  contour = contours[29]
  print(contour.shape)
  print(contour.reshape(-1).tolist())