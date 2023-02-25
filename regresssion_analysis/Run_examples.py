from helpful_scripts import *
from least_squares_funcs import *
from MLE_regression_funcs import *
import matplotlib.pyplot as plt
import numpy as np

# selected examples from ref. 1 in readme file.

x = np.matrix('5.133 10.124 15.060 19.946 24.899 29.792 \
              29.877 35.011 39.878 44.862 49.795')

y = np.matrix('0.265 0.287 0.282 0.286 0.310 0.333 0.343 \
              0.335 0.311 0.345 0.319')

example = 3

if example == 1:
    # =====================================
    # Example 15.3.1 in ref. 1 of readme file
    # perform univariate linear least squares regression.
    β = lsq_regress(y,x,1)
    print(f'β = \n{β}')
    
    # * For example 15.3.2 see chi-squared goodness of fit readme/ code.
    
    # Example 15.3.3
    # Find least sqare regression confidence intervals:
    N = 5000 # h = 1/N is step size in integrations for finding critical vals.
    β0_intvl, β1_intvl, σ_intvl =  linear_lsq_confidence(y,x,1,0.95, N)

if example == 2: 
    # =====================================
    # Example 15.4.1 in ref. 1 of readme file
    # perform univariate quadratic MLE regression, plot
    β = MLE_regress(y,x,2)[0]

    print(f'β = \n{β}')
    plot_regression(y,x,2)


if example == 3: 
    # =====================================
    # Here we find confidence intervals for the MLE approach.
    # then we test if βi = 0 using a t-test.
    # See Theorems 15.4.5 and 15.4.6
    # we again use Example 15.4.1 in ref. 1 in readme file

    N = 10000 # h = 1/N is step size in integrations for finding critical vals.
    β_intvls, σ2_intvl = MLE_confidence(y,x,2,0.95, N)

if example == 4:
    #=======================================
     # example 15.4.4
     # Find best polynomial fit (or at least the minimal
     # polynomial fit)
     α = 0.05
     p = 2
     poly_fit(x,y,α,p)






