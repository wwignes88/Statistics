import numpy as np
from distributions import *
from funcs import *
from helpful_scripts import *

# See exercise 11 of chapter 13 in ref. 1 (see readme file)
# We are testin for independence between observations.
O  = np.matrix(' 100 80 20;\
                 60  50 15;\
                 20  50  5')
α = 0.05

#- - - - - - - - 
def chi_test(O, α):
    DIM  = O.shape; W = DIM[1]; L = DIM[0] 
    v    = (L-1)*(W-1)             # degrees of freedom
    xc   = Percentile(0,1-α,Chi,v) # critical value
    
    CHI = 0  ; i = 0 
    N   = np.sum(O)
    while i < L:
        rowi= O[i,:]
        ni  = np.sum(rowi)
        pi  = ni/N
        
        j  = 0
        while j < W:
            nj  = np.sum(O[:,j])
            pj  = nj/N
            
            Oij = rowi[0,j];
            eij = N*pi*pj
            print('\n---------')
            print(f'p{j}  = {pj}')
            print(f'O{i}{j} = {Oij}')
            print(f'e{i}{j} = {eij}')           
            
            CHI += ((Oij - eij)**2)/eij
            j += 1
                
        i += 1
            
    if CHI <= xc:
        print(f' \n   Χ({v}) @ {1-α} = {np.round(xc,3)}  >  Χ = {np.round(CHI,3)} ')
        print(f'   cannot reject')
        TF = True
    if CHI > xc:
        print(f'   \nΧ({v}) @ {1-α} = {np.round(xc,3)}  <  Χ = {np.round(CHI,3)} ')
        print(f'   reject')    
        return False
    return TF

chi_test(O, α)





    





