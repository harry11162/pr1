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
		self.theta = np.array([0.09665107727050781, 0.9033489227294922])
		# self.theta = np.array([0.15, 0.75])
		# self.theta = np.array([0.5, 0.5])

		# baseline
		self.pos_mean = np.array([54.786153704471154, 88.05770594465166, 157.89083206704424])
		self.neg_mean = np.array([113.99712143126098, 114.73827898187346, 111.57334382005074])
		# 1e-2 brightness balance
		# self.pos_mean = np.array([48.86701685056011, 79.60973245683657, 142.8676232426787])
		# self.neg_mean = np.array([102.070563073621, 102.75075500990084, 99.78789801297734])

		# baseline
		self.pos_var = np.array([2858.435999679047, 2931.32427775746, 3225.6206572453143])
		self.neg_var = np.array([4652.383885100656, 4149.232742309653, 4574.364908847455])
		self.pos_cov = np.array([[2858.437879993419, 2569.127025950346, 1408.9920904072962], [2569.127025950346, 2931.326206020254, 2181.885322580114], [1408.9920904072962, 2181.885322580114, 3225.622779092941]])
		self.neg_cov = np.array([[4652.384212539043, 3978.274547250707, 3340.814303712031], [3978.274547250707, 4149.233034311849, 3838.148539083741], [3340.814303712031, 3838.148539083741, 4574.365230769311]])
		# 1e-2 brightness balance
		# self.pos_cov = np.array([[2307.7913213043944, 2130.218687344777, 1209.169643714179], [2130.218687344777, 2570.4753904449085, 2057.7593505854607], [1209.169643714179, 2057.7593505854607, 3157.2024510299384]])
		# self.neg_cov = np.array([[3718.032476570727, 3115.72782633423, 2498.3691936330083], [3115.72782633423, 3259.443526411473, 2960.6840173734363], [2498.3691936330083, 2960.6840173734363, 3658.4622824041353]])

		# 1
		# self.known_contour = np.array([219, 174, 175, 175, 173, 176, 171, 178, 171, 180, 169, 181, 168, 183, 168, 185, 166, 186, 164, 188, 164, 190, 162, 191, 162, 193, 160, 194, 159, 196, 159, 198, 157, 199, 155, 201, 155, 203, 153, 204, 153, 206, 157, 209, 160, 212, 164, 215, 167, 216, 169, 219, 171, 221, 173, 222, 173, 237, 175, 238, 175, 253, 177, 254, 177, 268, 179, 269, 179, 285, 180, 300, 182, 301, 182, 316, 184, 317, 184, 332, 186, 333, 186, 348, 188, 349, 189, 365, 189, 380, 191, 381, 191, 393, 272, 393, 277, 390, 280, 389, 282, 386, 284, 385, 289, 383, 290, 381, 293, 379, 298, 377, 299, 375, 302, 373, 307, 371, 310, 368, 314, 366, 316, 365, 318, 364, 319, 361, 323, 359, 325, 357, 325, 350, 327, 349, 327, 341, 329, 340, 329, 333, 331, 332, 331, 325, 332, 316, 334, 315, 334, 307, 336, 306, 336, 299, 338, 298, 338, 291, 340, 290, 340, 283, 342, 282, 342, 273, 343, 265, 345, 264, 345, 257, 347, 256, 347, 248, 349, 247, 349, 240, 351, 239, 351, 231, 352, 223, 354, 222, 354, 214, 356, 213, 356, 206, 358, 205, 358, 198, 360, 197, 360, 189, 362, 187, 359, 187, 353, 186, 352, 185, 348, 185, 343, 184, 342, 183, 338, 182, 333, 182, 328, 181, 323, 180, 322, 179, 317, 179, 313, 178, 312, 177, 307, 176, 301, 175, 296, 174])
		# 30
		self.known_contour = np.array([417, 83, 416, 85, 399, 85, 383, 86, 366, 87, 365, 89, 348, 90, 332, 91, 315, 91, 314, 93, 298, 94, 282, 94, 281, 96, 264, 97, 248, 98, 231, 98, 230, 100, 215, 100, 216, 119, 217, 138, 218, 156, 219, 175, 220, 194, 221, 212, 223, 215, 227, 217, 228, 219, 231, 220, 232, 222, 235, 223, 237, 225, 238, 233, 239, 244, 240, 255, 241, 266, 241, 278, 242, 288, 243, 289, 244, 300, 245, 311, 246, 322, 247, 333, 248, 344, 248, 354, 249, 365, 250, 377, 251, 388, 252, 399, 253, 410, 254, 421, 255, 426, 265, 436, 268, 439, 271, 440, 273, 443, 276, 445, 279, 446, 281, 448, 284, 451, 286, 452, 291, 458, 295, 460, 296, 462, 299, 465, 302, 467, 304, 469, 308, 470, 310, 469, 313, 466, 317, 465, 318, 463, 320, 463, 322, 462, 323, 460, 325, 459, 327, 459, 329, 458, 335, 454, 336, 452, 340, 451, 342, 450, 343, 448, 345, 447, 347, 447, 353, 443, 354, 441, 358, 440, 360, 439, 361, 437, 368, 435, 372, 432, 376, 429, 376, 422, 377, 412, 378, 401, 379, 392, 380, 382, 381, 373, 382, 372, 382, 362, 383, 352, 384, 343, 385, 342, 385, 332, 386, 322, 387, 313, 388, 312, 388, 303, 389, 292, 390, 283, 391, 273, 392, 262, 393, 253, 394, 243, 395, 233, 396, 223, 397, 217, 398, 214, 400, 212, 400, 210, 402, 208, 402, 206, 405, 204, 405, 202, 407, 200, 407, 198, 409, 196, 409, 194, 411, 192, 412, 189, 413, 186, 414, 181, 415, 164, 415, 147, 416, 131, 417, 130, 418, 114, 419, 96, 419, 83])
		self.known_contour = self.known_contour.reshape(-1, 1, 2)
		self.thr = 0.5
		self.area_lower_thr = 0.1
		self.area_higher_thr = 0.4
		self.known_contour = cv2.convexHull(self.known_contour)

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
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		h, w, _ = img.shape
		# YOUR CODE HERE
		
		# 1e-2 brightness balance
		# brightness = img * np.array([0.2126, 0.7152, 0.0722])[np.newaxis, np.newaxis, :]
		# brightness = brightness.sum(axis=2).mean()
		# img = img / (brightness * 1e-2)

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
		for i in contours:

			# epsilon = 0.01 * cv2.arcLength(i,True)
			# i = cv2.approxPolyDP(i,epsilon,True)

			i = cv2.convexHull(i)

			distance = cv2.matchShapes(i, self.known_contour, 1, 0.)
			area = cv2.contourArea(i)

			x,y,w2,h2 = cv2.boundingRect(i)
			EPS = 1e-5
			distance = (w2*h2 - area) / (area + EPS)

			if distance < self.thr and \
			area > h*self.area_lower_thr * w*self.area_lower_thr and \
			area < h*self.area_higher_thr * w*self.area_higher_thr:
				mat = np.zeros((h, w, 3))
				cv2.fillPoly(mat, pts=[i.reshape(-1, 2)], color=(255,255,255))

				# cv2.imshow('approx', mat)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()

				mat = mat[:, :, 0] > 0
				mat = label(mat)
				props = regionprops(mat)

				assert len(props) == 1
				bbox = props[0].bbox

				max_iou = 0.
				for b in boxes:
					max_iou = max(max_iou, self.get_iou(b, (bbox[1], bbox[0], bbox[3], bbox[2])))
				if max_iou < 0.5:
					boxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])

		return boxes



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