'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import label, regionprops
import pickle

class BinDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		self.theta = np.array([0.10346050262451172, 0.8965394973754883])
		self.pos_mean = np.array([52.386095804504784, 86.77734433811244, 160.51045693676554])
		self.neg_mean = np.array([114.72380872550686, 115.08867762774719, 110.91924750167485])
		self.pos_cov = np.array([[2686.252961657085, 2468.961241466471, 1379.5922080983273], [2468.961241466471, 2855.8629232274534, 2081.8482766584416], [1379.5922080983273, 2081.8482766584416, 2933.969079494998]])
		self.neg_cov = np.array([[4625.259638434907, 3971.790147196423, 3411.638614551249], [3971.790147196423, 4153.58882172896, 3887.183985549961], [3411.638614551249, 3887.183985549961, 4572.748169897826]])

		self.thr = 0.2
		self.area_lower_thr = 0.05
		self.area_higher_thr = 0.4

		with open("bin_detection/known_contours.pkl", "rb") as f:
			self.known_contours = pickle.load(f)

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
		# YOUR CODE HERE
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		h, w, _ = img.shape

		# squeeze the image into [H*W, 3]
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

		boxes = []
		for i in contours:
			# compute contour area, for later filtering
			area = cv2.contourArea(i)

			# compute similarity
			# specifically, the minimum distance from all the known contours
			distance = min([cv2.matchShapes(i, c, 3, 0.) for c in self.known_contours])

			if distance < self.thr and \
			area > h*self.area_lower_thr * w*self.area_lower_thr and \
			area < h*self.area_higher_thr * w*self.area_higher_thr:
				# use cv2.fillPoly and regionprops to compute bbox
				mat = np.zeros((h, w, 3))
				cv2.fillPoly(mat, pts=[i.reshape(-1, 2)], color=(255,255,255))

				mat = mat[:, :, 0] > 0
				mat = label(mat)
				props = regionprops(mat)

				assert len(props) == 1
				bbox = props[0].bbox

				# due to some reasons, cv2.findContours output 2 contours for each one contour
				# remove the other one using IoU
				max_iou = 0.
				for b in boxes:
					max_iou = max(max_iou, self.get_iou(b, (bbox[1], bbox[0], bbox[3], bbox[2])))
				if max_iou < 0.5:
					boxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])

		return boxes

	# compute IoU. It's a quite standard and easy algorithm, but it's tedious to implement.
	# so I simply copied this code from Internet.
	# https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
	def get_iou(self, bb1, bb2):
		bb1 = {'x1': bb1[0], 'y1': bb1[1], 'x2': bb1[2], 'y2': bb1[3]}
		bb2 = {'x1': bb2[0], 'y1': bb2[1], 'x2': bb2[2], 'y2': bb2[3]}
		# determine the coordinates of the intersection rectangle
		x_left = max(bb1['x1'], bb2['x1'])
		y_top = max(bb1['y1'], bb2['y1'])
		x_right = min(bb1['x2'], bb2['x2'])
		y_bottom = min(bb1['y2'], bb2['y2'])

		if x_right < x_left or y_bottom < y_top:
			return 0.0

		# The intersection of two axis-aligned bounding boxes is always an
		# axis-aligned bounding box
		intersection_area = (x_right - x_left) * (y_bottom - y_top)

		# compute the area of both AABBs
		bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
		bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
		assert iou >= 0.0
		assert iou <= 1.0
		return iou