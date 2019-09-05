import numpy as np

def tapped_x(x, window, k):
  """
  Take a long row vector and return mirrored sub row vector starting from k-window to k.
  Returns 0 for all negative index

  Inputs
  -------
  
  x : numpy array (row vector)
    input signal to feed filter
  window : integer
    sub x vector length.
  k : integer
    sub x vector last index

  Outputs
  -------
  
  tapped_x : mirrored sub row vector
  """
  prefixed_input = LMSUtils._range_x(x, k-window+1, k+1)
  return prefixed_input[::-1]

def dim_x(vec):
  """
  Take an vector and return number of rows or array length.

  Inputs
  -------

  vec : numpy array (row vector)

  Outputs
  -------

  dim : integer
    number of rows or array length.
  """

  return vec.shape[0]

@staticmethod
def conj(vec):
  """
  Take an complex vector and return its conjugate.

  Inputs
  -------

  vec : numpy array (row vector)

  Outputs
  -------

  conj_vec : numpy array (row vector)
    conjugated vector
  """
  return np.conjugate(vec)

def _x(x, index):
    if index < 0:
        if x.dtype == complex: return complex(0, 0)
        return 0;
    return x[index]

# Improve source code regularity
def _range_x(x, start, length):
    sub_x = [LMSUtils._x(x, start)]
    for it in range(start+1, length):
        sub_x = np.append(sub_x, LMSUtils._x(x, it))
    return sub_x