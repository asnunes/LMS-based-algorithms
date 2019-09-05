import numpy as np
from lms_base import *

class SignError(LMSBase):
  """
  Description
  ---------
  
  Implements the REAL Sign Error LMS-based algorithm for REAL valued data.
  (Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
  """
  @staticmethod
  def _coefficients_update_function(w_k, step, err_k, x_k):
    return w_k + 2 * step * np.sign(err_k) * x_k
