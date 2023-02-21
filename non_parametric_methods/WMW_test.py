

from helpful_scripts import *
import numpy as np
from tables import *


# Perform Wilcoxonn-Whitney-Man (WMW)test
# GFE is an array of True/ False values. Only one value should be set 
# to true at a time. First entry True tests whether x is stochastically
# smaller than y, i.e. P[x>a]<P[y>a] for all a. second entry tests if 
# x is stochastically greater than y. Third entry tests stochastic equivalence.

# X,Y are the data sets themselves (vectors in row matrix form)
# α is the test level. uα is the WMW critical value. At the moment 
# the WMW table in tables.py is incomplete so it is suggested to just
# look this value up and enter it into the code (hence why it is an input).
# However, for data sets less than 9 in length the function 'MWM_stat'
# imported from tables.py can read these values off.

# n is a counting variable used for the test for equality. 
# *call as 
#         WMW_test(GFE, X,Y,α, uα, 1) 
# or
#         WMW_test(GFE, X,Y,α, uα)
# attempting to call with other n values will cause a malfunction 
# for the equality test which will run the code a second time (until n==2) 
# with X,Y swapped. 

def WMW_test(GFE, X,Y,α, uα, n=1):
    
    
    x_stoch_smaller = GFE[0] # Test if x is stoch
    x_stoch_greater = GFE[1]
    x_equals_y      = GFE[2]
    
    if x_stoch_smaller:
        # swap X,Y - the test is then same as x_stoch_greater 
        xo = np.copy(Y)
        Y  = np.copy(X)
        X  = xo
    
    nx = X.shape[1] ; ny = Y.shape[1]
    
    # combine data
    XY = np.append(X,Y, axis = 1); 
    # rank data, taking average value of equal ranks.
    XY, XY_ranks, I_ = rank_avg(abs(XY)) ; N = XY.shape[1]
    
    
    # to perform WMW test, lets call all x vals 1 and all y vals 0
    y  = np.zeros(ny); y = np.matrix(y)
    x  = np.ones(nx) ; x = np.matrix(x)
    xy = np.append(x,y, axis = 1) # combine x,y
    xy = np.matmul(I_,xy.transpose()).transpose() # replicate ordering of X,Y ranks.
    
    print(f"xy [x=1's, y=0's]: \n{xy}")
    print(f'XY : \n{XY}')
    print(f'XY ranks: \n{XY_ranks}')
    
    
    # Find Wilcoxon rank sum (use 1 vals in xy as indices to sum x rank vals in XY)
    Wx = binary_indexed_sum(xy,XY_ranks)
    print(f'\nWx : {Wx}')
    
    # find WMW stat:
    Ux = Wx - nx*(nx+1)/2
    print(f'Ux : {Ux}')
    
    # find MWM statistic (see table.py)
    n  = np.min([nx,ny]) ; m = np.max([nx,ny])
    # MWM_stat(n,m,α ) # !!! table not updated past m=9! 
    # will need to look value up.
    
    
    if x_stoch_greater:
        if Ux <= uα:
            print(f'\nUx < uα = {uα} reject @ α = {α}.')
        if Ux >= uα:
            print(f'\nUx < uα = {uα} cannot reject @ α = {α}.')
    if x_stoch_smaller:
        Uy = Ux
        if Uy <= uα:
            print(f'\nUx < uα = {uα} reject @ α = {α}.')
        if Uy >= uα:
            print(f'\nUx < uα = {uα} cannot reject @ α = {α}.')
    if x_equals_y:  
        ux  = Ux # store Ux calculated from this loop
        # Now switch to testing for x_stoch_greater but with X,Y switched
        # will return what is Uy.
        GFE = [True, False, False]
        print(f'\n\n\n Running again....\n\n')
        uy = WMW_test(GFE, X,Y,α, uα, n=1)
        u  = np.min([ux,uy])
        print(f'ux,uy = {np.round([ux,uy],4)}')
        if u <= uα:
            print(f'\n{u}  <=  uα = {uα} \
                  reject @ α = {α}.')
        if u > uα:
            print(f'\n{u} >  uα - {uα}\
                  accept @ α = {α}.')
    
    E = nx*ny/2
    V = np.sqrt(nx*ny*(nx+ny+1)/12)
    z1_α = Percentile(-10,1-α/2,Standard_Normal,1)
    uα = E - z1_α*V
    print(f'\nlarge sample approximation: \nP({np.round(uα,3)}) = {uα}')
    
    return Ux



#============== Enter data ====================
# example 14.7.1
X = np.matrix(' 23 261 87 7 120 14 62 47 225 71 246 21') 
Y = np.matrix(' 55 320 56 104 220 239 47 246 176 182 33')
GFE = [True, False, False]
α  = .10; uα = 44 
#WMW_test(X,Y,α, uα)

# exercise 14.15
X = np.matrix(' 1067 919 1196 785 1126 936 918 1156 920 948 ') ; nx = X.shape[1]
Y = np.matrix(' 1105 1243 1204 1203 1310 1262 1234 1104 1303 1185'); ny = Y.shape[1]

GFE = [False, False, True]
α  = .10; uα = 27 
WMW_test(GFE, X,Y,α, uα)

