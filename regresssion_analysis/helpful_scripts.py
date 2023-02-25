import numpy as np
from distributions import *
from   sympy import * 
from   sympy.abc import x, y, z


# = = = = = = = = = = = = = = = = = =
# calculate the inverse of a mtarix
def inverse_(M):
    try:
        return np.linalg.inv(M)
    except:
        input("Singular Matrix, Inverse not possible.")



# = = = = = = = = = = = = = = = = = = 
# find v**2 where v is row matrix.
def square(v):
    return np.multiply(v,v)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# find critical x value (xc) where cumulative distribution
# equals Y where 0 <= Y <= 1.
def Percentile(N, x0, α, pdf,*argv):
    # h=1/N is step size. 
    h     = 1/N
    xi    = x0  ;  SUM   = 0
    while SUM <= α:
        f0   = pdf(xi,argv[0])
        fh   = pdf(xi+h,argv[0])
        SUM  += 0.5*(f0+fh)*h # trapezoidal integration
        xi   += h
    return xi

