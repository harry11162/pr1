'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import os, cv2
from roipoly import RoiPoly
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle

if __name__ == '__main__':

  folder = 'data/training'
  masks = []

  for i in range(14, 15):

    filename = '00{:02d}.jpg'.format(i)
    img = cv2.imread(os.path.join(folder,filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # display the image and use roipoly for labeling
    fig, ax = plt.subplots()
    ax.imshow(img)
    my_roi = RoiPoly(fig=fig, ax=ax, color='r')
    
    # get the image mask
    mask = my_roi.get_mask(img)
    
    with open("mask_00{:02d}.pkl".format(i), "wb") as f:
      pickle.dump(mask, f)

