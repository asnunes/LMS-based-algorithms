import numpy as np
import lms_utils

class LMSBase:
    """
    Description
    ---------
    Author: Alexandre Henrique da Silva Nunes
    Based on Matlab code by Markus Vinícius Santos Lima available at 
    https://www.mathworks.com/matlabcentral/fileexchange/3582-adaptive-filtering
    
    Methods
    ---------
    fit :
        update filter params based on a desired signal and inputs.
    """
    
    def __init__(self, filter_order_num, initial_coefficients):
        """
        Inputs
        -------
        
        filter_order_num : int
                    Order of the FIR filter
        initial_coefficients : numpy array (collumn vector)
                    initial filter coefficients
                
                    
        Variables
        --------
        num_of_coefficients : int
            FIR filter number of coefficients.
        
        errors_vector : numpy array
            FIR error vectors. error_vector[k] represents the output erros at iteration k.
        outputs_vector : numpy array (collumn vector)
            Store the estimated output of each iteration. outputs_vector[k] represents the output erros at iteration k
        coefficients_mtx : numpy array
            Store the estimated coefficients for each iteration. (Coefficients at one iteration are COLUMN vector)
        """
        
        self.filter_order_num = filter_order_num
        self.num_of_coefficients = self.filter_order_num + 1
        
        self.errors_vector = np.array([0], dtype=initial_coefficients.dtype)
        self.outputs_vector = np.array([0], dtype=initial_coefficients.dtype)
        self.coefficients_mtx = np.array(initial_coefficients, dtype=initial_coefficients.dtype)
        
    def fit(self, desired, x, step):
        """
        Fit filter parameters to considering desired vector and input x. desired and x must have length K,
        where K is the number of iterations
        
        Inputs
        -------
        
        desired : numpy array (row vector)
            desired signal
        x : numpy array (row vector)
            input signal to feed filter
        step : Convergence (relaxation) factor.
        
        Outputs
        -------
        
        python dic :
            outputs : numpy array (collumn vector)
                Store the estimated output of each iteration. outputs_vector[k] represents the output erros at iteration k
            errors : numpy array (collumn vector)
                FIR error vectors. error_vector[k] represents the output erros at iteration k.
            coefficients_mtx : numpy array
                Store the estimated coefficients for each iteration. (Coefficients at one iteration are COLUMN vector)      
        """
        
        k_max = lms_utils.dim_x(desired)
        self._initialize_vars(k_max)
        
        for k in range(k_max):
            x_k = lms_utils.tapped_x(x, self.num_of_coefficients, k)
            w_k = self.coefficients_mtx[:, k]
            y_k = np.dot(lms_utils.conj(w_k), x_k)
            err_k = desired[k] - y_k
        
            next_w_k = self._coefficients_update_function(w_k, step, err_k, x_k)
            
            self._update(err_k, y_k, next_w_k, k)
            
        return {
            'outputs': self.outputs_vector,
            'errors': self.errors_vector,
            'coefficients': self.coefficients_mtx
        }

    @staticmethod
    def _coefficients_update_function(w_k, step, err_k, x_k):
      # Override this method
      return 0
      return w_k + step * lms_utils.conj(err_k) * x_k
        
    def _initialize_vars(self, k_max):
        self.errors_vector = np.zeros((k_max, 1))
        self.outputs_vector = np.zeros((k_max, 1), dtype=self.outputs_vector.dtype)
        
        initial_coefficients = self.coefficients_mtx[:, [0]]
        last_coefficients = np.zeros((self.num_of_coefficients, k_max), dtype=initial_coefficients.dtype)
        self.coefficients_mtx = np.append(initial_coefficients, last_coefficients, axis=1)
    
    def _update(self, err, y, next_w, k):
        self.errors_vector[k, 0] = err
        self.outputs_vector[k, 0] = y
        self.coefficients_mtx[:, k+1] = np.transpose(next_w)
