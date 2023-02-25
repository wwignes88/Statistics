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
def MLE_regress(y,x,x_pow):
    n = x.shape[1];  p = x_pow
    
    # initialize X matrix.
    ones = np.matrix(np.ones(x.shape[1])).transpose()
    X    = ones
    
    # hstack higher powers of x for higher degree polynomial fits:
    # yi = β0 + β1xi + β2xi**2 + ...
    i = 1; xi = x;
    while i <= x_pow:
        X  = np.hstack([X,xi.transpose()])
        xi = np.multiply(xi,x)
        i += 1

    Y  = y.transpose()
    XX = inverse_(np.matmul(X.transpose(),X))
    XY = np.matmul(X.transpose(),Y)
    β  = np.matmul(XX,XY)
    
    # Find covariance matrix:
    A = XX
    
    # Find σ**2:
    Xβ = np.matmul(X,β)
    σ2 = np.matmul((Y-Xβ).transpose(),Y-Xβ)/(n-p-1)
    σ2 = σ2[0,0]
    return β, Y, X, σ2, A



# = = = = = = = = = = = = = = = = = = 
# plot least squares regress/ find confidence intervals
# for β0, β1, σ**2
# based on LINEAR least squares regression:
#        yi = β0 + β1xi 
# *Note: x,y presumed vector (row matrix) inputs.
def MLE_confidence(y,x,x_pow,γ,N):
    
    n = x.shape[1]; p = x_pow

    # find t_(1-γ/2) percentile (students t distribution)
    x0,N = -15, 1000
    t_plus_γ2  = Percentile(N,x0,(1+γ)/2,student_t,n-2)
    t_test_βi_zer0  = Percentile(N,x0,0.975,student_t,n-p-2)
    #print(f'tγ = {t_test_βi_zer0}\n')

    # perform least squares linear regression,
    β, Y, X, σ2, A = MLE_regress(y,x,x_pow) 
    
    # Find confidence interval for βi
    i = 0; β_intvls = []
    while i < β.shape[0]:
        aii   =  A[i,i]
        βi    =  β[i,0]
        #print(f'β{i} = {βi}')
        
        left  =  βi - t_plus_γ2*np.sqrt(σ2*aii)
        right =  βi + t_plus_γ2*np.sqrt(σ2*aii)
        β_intvls.append([left,right])
        
        # Test if βi = 0:
        ti = βi/np.sqrt(aii*σ2)
        #print(f'   t{i} = {ti}')
        if abs(ti) < t_test_βi_zer0:
            print(f'   β{i} ~ 0')
        
        i  += 1
    
    
    # Find σ2 confidence interval:
    x0,N = 0.0001, 1000
    χ1_minus_γ2 =  Percentile(N,x0,(1-γ)/2,Chi_,n-2)
    χ1_plus_γ2  =  Percentile(N,x0,(1+γ)/2,Chi_,n-2)
        
    σ2_left  = (n-p-1)*σ2/χ1_plus_γ2
    σ2_right = (n-p-1)*σ2/χ1_minus_γ2
    σ2_intvl = [σ2_left,σ2_right]

    print(f'\nβ  confidence intvls: \n{np.round(β_intvls, 5)}')
    print(f'\nσ**2  confidence intvl: {np.round(σ2_intvl, 5)}')
    
    #print(f'\nt_({(1+γ)/2})({n-2})   = {np.round(t_plus_γ2,4)}')
    #print(f'χ_({(1+γ)/2})({n-2})   = {np.round(χ1_plus_γ2,4)}')
    #print(f'χ_({np.round((1-γ)/2,3)})({n-2})   = {np.round(χ1_minus_γ2,4)}')
    
    
    return β_intvls, σ2_intvl



# = = = = = = = = = = = = = = = = = = 
# plot least squares regression data (x,y) and regression line:
#       yi = β0 + β1xi + β2xi**2 + ...
# y,x are input data. x_pow is the degree of the (univariate)
# polynomial fit.

# *MLE and least squares regression have the same
# y value, so we can use either one to plot.

def plot_regression(y,x,x_pow):
    
    # perform least squares linear regression,
    β = MLE_regress(y,x,x_pow)[0] 
    
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
    Δ = xmax-xmin  ; h = Δ/100
    smin = xmin - 0.1*Δ ; smax = xmax + 0.1*Δ
    s = np.arange(smin,smax,0.5)
    y_line = y_func(s)
    
    plt.plot(x,y, '*', s,y_line)

    
    
#================================================


# x.f pg 525 and pg. 516. This is the expression S(β)
# inputs x,y,β are all COLUMN matrix/ vectors.
def sβ(X,Y,β):
    Xβ   = np.matmul(X,β)
    Y_Xβ = Y - Xβ
    S    = np.matmul(Y_Xβ.transpose(),Y_Xβ)
    return S[0,0]


# Use GLR test to find the best fit for the univariate
# polynomial yi = yi = β0 +  β1*xi + ... + βp*xi^p
# i.e. we are testing Ho: βm = ... = βp = 0.
def poly_fit(x,y,α, p):
    n = x.shape[1]

    #----------------------------------------
    
    # perform regression on 
    #     yi = β0 +  β1*xi + ... + βp*xi^p
    β, Y, X, σ2, A = MLE_regress(y,x,p)

    # Find Sβ  =  (Y-Xβ)'(Y-Xβ)
    Sp = sβ(X,Y,β) 
    print(f'Sβ(p={p}) = {np.round(Sp,6)}')
    
    m  = p ; Sm = Sp
    while m > 0:
        # perform regression on yi, where
        #     yi = Xβ^ = β0^ +  β1^*xi + ... + βp^*xi**{m-1}
        β, Y, X, σ2, A = MLE_regress(y,x,m-1)

        # Find Sβ(m)  =  (Y-Xβ^)'(Y-Xβ^)
        Sm    = sβ(X,Y,β) 
        print(f'Sβ(m={m-1}) = {np.round(Sm,4)}')

        # f statistic:
        num  = (Sm-Sp)/(p-m+1)  ; den = Sp/(n-p-1)
        f    = num/den
        
        # find critical value
        args = p-m+1, n-p-1 ; N = 10000
        f1_α = Percentile(N, 1/(N), 1-α, Snedecor,args)
        print(f'f1_α = {np.round(f1_α,6)}')
        print(f'f    = {np.round(f,6)}')
        if f > f1_α:
            
            print(f'   reject β{m} : β{p} = 0\n')

        if f <= f1_α:
            print(f'  accept β{m} : β{p} = 0\n')
        
        if m > 0:
            plot_regression(y,x,m)

        m += -1















