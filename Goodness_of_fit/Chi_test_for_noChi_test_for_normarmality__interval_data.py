import numpy as np
import pandas as pd
from multinomial_MLEs import *
from raphson import *
from funcs import *

# Example 13.7.3 from 'Introduction to Probability and Mathematical Statistics'
# by Bain and Englehardt (2nd ed.)

PLOT    = False # switch to True for plotting.
RAPHSON = False  # switch to True to estimate μ, σ  with Newton-Raphson

# First run with PLOT = True. Estimate μ0, σ0.
# Now run a second time with RAPHSON set to True to 
# get a more accurate estimate of μ, σ.
# estimate μ, σ based on plot:
μ, σ  = -20 ,10 # initial guess based on plot (Raphson set to False)
                 # these are the values Raphson should arrive at

if RAPHSON: # update μ, σ with output from Raphson.
    μ, σ   =  -19.867115036443728, 11.2626087946973


#============= enter data ====================
# create pandas data frame [not necessary but its good to get in the habit]
dict = {'intervals':[(-80,-70),(-70,-60),(-60,-50),(-50,-40),(-40,-30),\
                     (-30,-20),(-20,-10),(-10,0),(0,10),(10,20),(20,30)],
        'O': [1,2,2,2,8,24,26,11,2,1,1]}    
df     = pd.DataFrame(dict)     

# O is for observed values in each interval
# intvls is for velocity intervals in which observations were made.
O    = df['O'].values  ; intvls = df['intervals'].values 
N    = np.sum(O) ; c = len(O) ;  k = 2 # c = # cells. k = # estimated parameters.


#===================================================
# Calculate [integrate over intervals] probabilities:

print(f'\nProbabilities:')
P = np.array([])
i = 0 
while i < c:
    intvli = intvls[i]
    ai   = intvli[0]
    ai_1 = intvli[1]
    zi   = (ai   - μ)/σ
    zi_1 = (ai_1 - μ)/σ
    pi = Φ(zi,zi_1)
    P  = np.append(P,pi)
    i += 1
print(f'\nP: {np.round(P,3)}') 

# pool data according to the criteria pi > 5/N:
Pgrouped = np.array([]) 
Ogrouped = np.array([])
Grouped_intvls = []

i   =  0 
while i < c:
    ai_lower = intvls[i][0]
    oi = O[i] ; print(f'\no{i} = {oi}')
    pi = P[i]
    print(f'sum p {i+1}: {np.sum(P[i+1:])}')
    if pi < 5/N or np.sum(P[i+1:]) < 5/N :
        j = i+1 ;  pool = True ; 
        while pool and j < c:
            oi  += O[j]        # pool observations
            pi  += P[j]        # pool probabilities
            print(f'pooled p{i}, p{j} ; o{j} = {O[j]}')
            if pi > 5/N and np.sum(P[j+1:]) > 5/N:
                pool = False
            j += 1
        i = j-1
    ai_upper = intvls[i][1]
    Grouped_intvls.append([ai_lower,ai_upper])
    Pgrouped = np.append(Pgrouped,pi)
    Ogrouped = np.append(Ogrouped,oi)
    i += 1

print(f'\nGrouped_intvls: \n{Grouped_intvls}')
print(f'Ogrouped: \n{Ogrouped}')
print(f'Pgrouped: \n{Pgrouped}')
ARGS = Ogrouped, Grouped_intvls 


#=============================================
# calculate degrees of freedom:
    
c = len(Ogrouped) # updated number of cells.

# alternative: see read me file.
v = c-1-k
print(f'\nv = {v}')


#===========================================
# Newton-raphons (approximate σ, μ with MLE)
# We have two functions to iterate over; fμ and fσ (see equations 6.7 of read me file)
# These functions are imported from 'multinomial_MLEs.py'

if RAPHSON:
    σ, μ = newton_raphson_2D(fμ , fσ,  μ, σ, ARGS)
    
#========= PLOTTING fμ ========================
# Because Newton-Raphson is very susceptible to 
# the initial guess, it is good to plot at least
# one of the two functions in question to get a feel
# for where it approaches zero. Here we plot fμ (which 
# is a function of μ and μ and σ).

if PLOT:
    grid_view = 15,20 # toggle grid view
    N_grid    = 2 # number of grid points
    PLOT_MLEs(N_grid,grid_view, ARGS )
# Note: due to iterations and integrations, calculating the function 
# takes time, so anything higher then N=5 will take over a half-minute.

#====================================================
# Calculate expectation values/ CHI square statistic:
CHI =  0 
i = 0
while i < len(Pgrouped):
    oi = Ogrouped[i] ;# print(f'\no{i} = {oi}')
    pi = Pgrouped[i] 
    ei = pi*N # expected value for ith interval.
    #print(f'e{i} = {ei}')
    CHI += ((oi - ei)**2)/ei # Chi statistic
    i += 1
    
#===========================================
# Perform Chi-squared goodness of fite test.
α  = 0.05 # set test percentile.
# note that for low degress of freedom Chi squared distribution
# is sharply peaked. Ergo, Percentile function is set with a very small
# step size (see 'funcs.py')
xc = Percentile(0.000000001,1-α,Chi,v) # critical value

if CHI <= xc:
    print(f' \n   Χ({v}) @ {1-α} = {np.round(xc,3)}  >  Χ = {np.round(CHI,3)} ')
    print(f'   cannot reject')

if CHI > xc:
    print(f'   \nΧ({v}) @ {1-α} = {np.round(xc,3)}  <  Χ = {np.round(CHI,3)} ')
    print(f'   reject')



