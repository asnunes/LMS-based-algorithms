import lms_utils
import numpy as np
from lms_base import *

class ComplexLMS(LMSBase):
  """
    Description
    ---------
    
    Implements the Complex LMS algorithm for COMPLEX valued data.
    (Algorithm 3.2 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
  """
  @staticmethod
  def _coefficients_update_function(w_k, step, err_k, x_k):
    return w_k + 2 * step * lms_utils.conj(err_k) * x_k