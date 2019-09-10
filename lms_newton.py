import numpy as np
from lms_base import *

class LMSNewton(LMSBase):
  """
  Description
  ---------
  
  Implements the REAL LSM-Newton LMS-based algorithm for REAL valued data.
  (Algorithm 4.2 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
  """
  def __init__(self, initial_inv_estimate_R=None, *args, **kargs):
    """
    Description
    ---------
    Extends original base LMS class just adding new initial_inv_estimate_R param

    Input
    ---------
    initial_inv_estimate_R: numpy array
      Numpy array NxN where N is the number of filter coefficients. It's is the estimate 
      inverse correlation matrix. Default value is None and it's set as an eye matrix for the
      first iteration.
    """
    super(LMSBase, self).__init__(*args, **kargs)
    self.inv_estimate_R = initial_inv_estimate_R or np.eye(self.num_of_coefficients)

  def _coefficients_update_function(self, w_k, step, err_k, x_k, **kargs):
    """
    Input
    ---------
    kargs alpha : float number
      Usually between 0 and 0.1. Used to estimate the correlation matrix and its inverse.
    """
    alpha = kargs['alpha']
    prod_a = np.dot(self.inv_estimate_R, np.transpose(np.array([x_k])))
    prod_b = np.dot(np.transpose(np.conj(x_k)), self.inv_estimate_R)
    num = np.dot(prod_a, np.array([prod_b]))
    den = (1 - alpha)/alpha + np.dot(np.dot(np.dot(np.conj(x_k), self.inv_estimate_R), x_k))
                     
    next_w_k = w_k + step * err_k * np.dot(self.inv_estimate_R, x_k)
    self.update_inv_estimate_R(alpha, num, den)

    return next_w_k

  def update_inv_estimate_R(self, alpha, num, den):
    self.inv_estimate_R = 1/(1-alpha)*(self.inv_estimate_R - num/den) 