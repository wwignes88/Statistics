
from helpful_scripts import *
import numpy as np

# c.f. theorem 14.8.1 in ref. 1 of readme file
# we find both Pearson and Spearmans correlation coefficient 
# and their associated T statistics. We test
#        Ho: F(x,y) = F(x)F(y)  -->  independence

def Spearmans_rank_correlation(X,Y):
    nx = X.shape[1] ;  ny = Y.shape[1] ; n = nx
    
    #================================
    print(f'Pearson: ')
    # Find r, t (Pearsons correlation)
    x_avg = np.sum(X[0])/nx
    y_avg = np.sum(Y[0])/ny
    
    X2 = np.sum( np.multiply(X,X) )
    Y2 = np.sum( np.multiply(Y,Y) )
    XY = np.sum( np.multiply(X,Y) )
    num = XY - n*x_avg*y_avg
    
    A   = np.sum(X2)-n*x_avg**2
    B   = np.sum(Y2)-n*y_avg**2
    den = np.sqrt(A*B)
    r = num/den
    print(f'r: {np.round(r,3)}')
    t = np.sqrt(n-2)*r/np.sqrt(1-r**2)
    print(f't: {np.round(t,3)}')
    
    # find critical t value
    t1_α = 1.860 # t1_α(b-2)
    
    if t >= -t1_α:
        print(f't = {np.round(t,4)} >= -t1_α = {np.round(-t1_α,4)}   :  reject.')

    if t < -t1_α:
        print(f't = {np.round(t,4)} < -t1_α = {np.round(-t1_α,4)}   :   accept.')
    
    
    #==================================
    print(f'\n-----------\nSpearman: ')
    # find R, T (Spearman's correlation)
    # sort vals in ascending order, assign ranks.
    # sort vals in [absolute] ascending order, assign ranks [ranks not averaged].
    x, x_ranks, Ix = rank_avg(X) 
    y, y_ranks, Iy = rank_avg(Y) 

    # restore x_ranks according to the original ordering of y.
    x_ranks = np.matmul(inverse_(Ix),x_ranks.transpose()).transpose()
    # now shuffle x_ranks in whatever fashion y was shuffled to create y_ranks.
    x_ranks = np.matmul(Iy,x_ranks.transpose()).transpose()
    #print(f'rank y: \n{y_ranks}')
    #print(f'rank x: \n{x_ranks}')
    
    # calculate Spearman's correlation coefficient R
    d = y_ranks - x_ranks
    #print(f'd: \n{d}')
    
    D2 = np.sum(np.multiply(d,d))
    R  = 1-6*D2/(n*(n**2-1))
    print(f'R: {np.round(R,3)}')

    T = np.sqrt(n-2)*R/np.sqrt(1-R**2)
    print(f'T: {np.round(T,3)}')
    
    # find critical t value
    t1_α = 1.860 # t1_α(b-2)
    
    if t >= -t1_α:
        print(f't = {np.round(t,4)} >= -t1_α = {np.round(-t1_α,4)}   :   reject.')

    if t < -t1_α:
        print(f't = {np.round(t,4)} < -t1_α = {np.round(-t1_α,4)}   :   accept.')
        
        
        
        
        
        
#=============== Enter data ================

X = np.array([10.8,12.7,13.9,18.1,19.4,21.3,23.5,24.0,24.6,25.0,\
              25.4,27.7,30.1,30.6,32.3,33.3,34.7,38.8,40.3,55.5]) 
Y = np.array([9.8,13.0,10.7,19.2,18.0,20.1,20.0,21.2,21.3,25.5,\
              25.7,26.4,24.5,27.5,25.0,28.0,37.4,43.8,35.8,60.9]) 
X = np.matrix(X)  ;   Y = np.matrix(Y)  
#Spearmans_rank_correlation(X,Y)


#-------------------------------
print(f'\nEXERCISE 14.17:\n')
x   = np.matrix(' 23 20 26 25 48 26 25 24 15 20')
y   = np.matrix(' 20 30 16 33 23 24  8 21 13 18')
Spearmans_rank_correlation(x,y)


