import numpy as np
import matplotlib.pyplot as plt
from distributions import *
from helpful_scripts import *
from   sympy import * 
from   sympy.abc import x, y, z




# = = = = = = = = = = = = = = = = = = 
# Perform least squares regression:
#   β  = (X'X)^{-1}X'Y
#   yi = β0 + β1xi + β2xi**2 + ...
# x_pow is highest power of xi
def lsq_regress(y,x,x_pow):
    
    # initialize X matrix.
    ones = np.matrix(np.ones(x.shape[1])).transpose()
    X    = np.hstack([ones,x.transpose()])
    
    # hstack higher powers of x for higher degree polynomial fits:
    # yi = β0 + β1xi + β2xi**2 + ...
    i = 1; xi = x;
    while i < x_pow:
        xi = np.multiply(xi,x)
        X    = np.hstack([X,xi.transpose()])
        i += 1

    Y  = y.transpose()
    XX = inverse_(np.matmul(X.transpose(),X))
    XY = np.matmul(X.transpose(),Y)
    β  = np.matmul(XX,XY)
    return β

# = = = = = = = = = = = = = = = = = = 
# plot least squares regression data (x,y) and regression line:
#       yi = β0 + β1xi + β2xi**2 + ...
# y,x are input data. x_pow is the degree of the (univariate)
# polynomial fit.
def plot_lsq_regression(y,x,x_pow):
    
    # perform least squares linear regression,
    β = lsq_regress(y,x,x_pow)  
    
    # define β0:
    β0  = β[0,0] ; y_str = f'{β0}'
    i = 1 ; 
    # *Note: we use 'z' in the string because
    # A) sympy recognizes x,y,z, B) we already 
    # used x,y for other purposes.
    while i < β.shape[0]:
        βi     = β[i,0]
        y_str += f' + {βi}*z**{i}'
        i     += 1
    #print(f'y = {y_str}')
    
    y_func = lambdify(z,y_str)
    # Plot regression/ data
    xmin = np.min(x) ; xmax = np.max(x)
    Δ = xmax-xmin
    smin = xmin - 0.1*Δ ; smax = xmax + 0.1*Δ
    s = np.arange(smin,smax,0.5)
    y_line = y_func(s)
    plt.figure()
    plt.plot(x,y, '*', s,y_line)
    
    



# = = = = = = = = = = = = = = = = = = 
# plot least squares regress/ find confidence intervals
# for β0, β1, σ**2
# based on LINEAR least squares regression:
#        yi = β0 + β1xi 
# *Note: x,y presumed vector (row matrix) inputs.
def linear_lsq_confidence(y,x,x_pow,γ, N):
    n = x.shape[1]
    # perform least squares linear regression,
    β = lsq_regress(y,x,x_pow)  
    β0, β1 = β[:,0]
    
    
    # Calculate SSE, σ
    SSE = np.sum(square(y - β0 - β1*x))
    σ2 = SSE/(n-2)
    σ  = np.sqrt(σ2)
    #print("\nSSE  = ", SSE)
    #print("\nσ**2 = ", σ2 )
    #print("σ    = ", np.sqrt(σ2) )
    #print("β1   = ", β1 )
    #print("β0   = ", β0 )


    #======================================================
    # find confidence intervals of least square regression
    # β0, β1 are inputs obtained from performing regression
    # γ is arbitrary - sets the test level for the t statistic.
        
    # find t_(1-γ/2) percentile (students t distribution)
    x0 = -15
    t_plus_γ2  = Percentile(N,x0,(1+γ)/2,student_t,n-2)

    # β0 confidence interval
    x_avg = np.sum(x)/n
    X2 = np.sum(square(x))
    a = np.sum(square(x-x_avg))
    β0_left  = β0 - t_plus_γ2*σ*np.sqrt(X2/(n*a))
    β0_right = β0 + t_plus_γ2*σ*np.sqrt(X2/(n*a))
    β0_intvl = [β0_left, β0_right]
    
    # β1 confidence interval
    a = np.sum(square(x-x_avg))
    β1_left  = β1 - t_plus_γ2*σ/np.sqrt(a)
    β1_right = β1 + t_plus_γ2*σ/np.sqrt(a)
    β1_intvl = [β1_left, β1_right]

    # σ confidence interval
    x0 = 0.0001
    χ1_minus_γ2 =  Percentile(N,x0,(1-γ)/2,Chi_,n-2)
    χ1_plus_γ2  =  Percentile(N,x0,(1+γ)/2,Chi_,n-2)
    σ_left  = (n-2)*σ2/χ1_minus_γ2
    σ_right = (n-2)*σ2/χ1_plus_γ2
    σ_intvl = [σ_left, σ_right]

    #print(f'\nβ0 confidence intvl: {np.round(β0_intvl, 5)}')
    #print(f'β1 confidence intvl: {np.round(β1_intvl, 5)}')
    #print(f'σ**2  confidence intvl: {np.round(σ_intvl, 5)}')
    
    #print(f'\nt_({(1+γ)/2})({n-2})   = {np.round(t_plus_γ2,4)}')
    #print(f'χ_({(1+γ)/2})({n-2})   = {np.round(χ1_plus_γ2,4)}')
    #print(f'χ_({np.round((1-γ)/2,3)})({n-2})   = {np.round(χ1_minus_γ2,4)}')
    
    
    return β0_intvl, β1_intvl, σ_intvl



