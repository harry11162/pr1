'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import label, regionprops

class BinDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		# with resize
		# self.theta = np.array([0.09665107727050781, 0.9033489227294922])
		self.theta = np.array([0.5, 0.5])
		self.pos_mean = np.array([54.786153704471154, 88.05770594465166, 157.89083206704424])
		self.neg_mean = np.array([113.99712143126098, 114.73827898187346, 111.57334382005074])
		self.pos_var = np.array([2858.435999679047, 2931.32427775746, 3225.6206572453143])
		self.neg_var = np.array([4652.383885100656, 4149.232742309653, 4574.364908847455])
		self.pos_cov = np.array([[2858.437879993419, 2569.127025950346, 1408.9920904072962], [2569.127025950346, 2931.326206020254, 2181.885322580114], [1408.9920904072962, 2181.885322580114, 3225.622779092941]])
		self.neg_cov = np.array([[4652.384212539043, 3978.274547250707, 3340.814303712031], [3978.274547250707, 4149.233034311849, 3838.148539083741], [3340.814303712031, 3838.148539083741, 4574.365230769311]])

		self.known_contour = np.array([219, 174, 175, 175, 173, 176, 171, 178, 171, 180, 169, 181, 168, 183, 168, 185, 166, 186, 164, 188, 164, 190, 162, 191, 162, 193, 160, 194, 159, 196, 159, 198, 157, 199, 155, 201, 155, 203, 153, 204, 153, 206, 157, 209, 160, 212, 164, 215, 167, 216, 169, 219, 171, 221, 173, 222, 173, 237, 175, 238, 175, 253, 177, 254, 177, 268, 179, 269, 179, 285, 180, 300, 182, 301, 182, 316, 184, 317, 184, 332, 186, 333, 186, 348, 188, 349, 189, 365, 189, 380, 191, 381, 191, 393, 272, 393, 277, 390, 280, 389, 282, 386, 284, 385, 289, 383, 290, 381, 293, 379, 298, 377, 299, 375, 302, 373, 307, 371, 310, 368, 314, 366, 316, 365, 318, 364, 319, 361, 323, 359, 325, 357, 325, 350, 327, 349, 327, 341, 329, 340, 329, 333, 331, 332, 331, 325, 332, 316, 334, 315, 334, 307, 336, 306, 336, 299, 338, 298, 338, 291, 340, 290, 340, 283, 342, 282, 342, 273, 343, 265, 345, 264, 345, 257, 347, 256, 347, 248, 349, 247, 349, 240, 351, 239, 351, 231, 352, 223, 354, 222, 354, 214, 356, 213, 356, 206, 358, 205, 358, 198, 360, 197, 360, 189, 362, 187, 359, 187, 353, 186, 352, 185, 348, 185, 343, 184, 342, 183, 338, 182, 333, 182, 328, 181, 323, 180, 322, 179, 317, 179, 313, 178, 312, 177, 307, 176, 301, 175, 296, 174])
		self.known_contour = self.known_contour.reshape(120, 1, 2)
		self.thr = 0.2
		self.area_thr = 0.05

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		h, w, _ = img.shape
		# YOUR CODE HERE
		# pos = ((img - self.pos_mean[np.newaxis, np.newaxis, :]) ** 2) / self.pos_var[np.newaxis, np.newaxis, :]
		# pos += np.log(self.pos_var)[np.newaxis, np.newaxis, :]
		# pos = pos.sum(axis=2)
		# pos -= 2 * np.log(self.theta[0])

		# neg = ((img - self.neg_mean[np.newaxis, np.newaxis, :]) ** 2) / self.neg_var[np.newaxis, np.newaxis, :]
		# neg += np.log(self.neg_var)[np.newaxis, np.newaxis, :]
		# neg = neg.sum(axis=2)
		# neg -= 2 * np.log(self.theta[1])

		img = img.reshape(-1, 3)

		dis = img - self.pos_mean[np.newaxis, :]  # (N, 3)
		pos = np.matmul(dis, np.linalg.inv(self.pos_cov))
		pos = (pos * dis).sum(axis=1)
		pos = pos + np.log(np.linalg.det(self.pos_cov))
		pos = pos - 2 * np.log(self.theta[0])

		dis = img - self.neg_mean[np.newaxis, :]  # (N, 3)
		neg = np.matmul(dis, np.linalg.inv(self.neg_cov))
		neg = (neg * dis).sum(axis=1)
		neg = neg + np.log(np.linalg.det(self.neg_cov))
		neg = neg - 2 * np.log(self.theta[1])

		mask_img = (pos < neg).astype(int).reshape(h, w)
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		# YOUR CODE HERE
		h, w = img.shape

		contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_FLOODFILL, method=cv2.CHAIN_APPROX_TC89_L1)
		img = img.astype(np.uint8) * 255
		img = np.stack([img] * 3, axis=2)

		boxes = []
		contours_to_draw = []
		for i in contours:
			distance = cv2.matchShapes(i, self.known_contour, 1, 0.)
			area = cv2.contourArea(i)
			if distance < self.thr and area > h*self.area_thr * w*self.area_thr:
				mat = np.zeros((h, w, 3))
				cv2.fillPoly(mat, pts=[i.reshape(-1, 2)], color=(255,255,255))

				mat = mat[:, :, 0] > 0
				mat = label(mat, 8)
				props = regionprops(mat)

				bbox = props[0].bbox
				boxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])

		return boxes


