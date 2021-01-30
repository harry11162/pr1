'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    self.theta = np.array([0.36599892, 0.3245804, 0.30942068])
    self.mu = np.array([[0.75250609, 0.34808562, 0.34891229],
                        [0.35060917, 0.73551489, 0.32949353],
                        [0.34735903, 0.33111351, 0.73526495]])
    self.sigma2 = np.array([[0.03705927, 0.06196869, 0.06202255],
                            [0.05573463, 0.03478593, 0.05602188],
                            [0.05453762, 0.05683331, 0.03574061]])
	
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    # YOUR CODE HERE
    # Just a random classifier for now
    # Replace this with your own approach 
    prior = -2 * np.log(self.theta)  # (3,)
    loglikelihood = (X[:, np.newaxis, :] - self.mu[np.newaxis, :, :]) ** 2 / self.sigma2[np.newaxis, :, :] \
                      + np.log(self.sigma2)[np.newaxis, :, :]
    joint_loglikelihood = loglikelihood.sum(axis=2) + prior[np.newaxis, :]

    y = np.argmin(joint_loglikelihood, axis=1) + 1
    print(y)
    return y
