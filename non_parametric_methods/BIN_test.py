# *Also referred to as 'signed test' in ref. 1 of readme file
# see pgs 404-405 and beginning of ch. 14 in ref. 1 of readme file.


import numpy as np
from helpful_scripts import BIN_CDF, Percentile, Standard_Normal

# input: GLE is true/ false array 
#        n, T are the number of observations and number of negative differences,
#        respecitvely. These inputs are for cases in which the number of success/ 
#        failures are already known and need not be calculated. 
#       *IMPORTANT: If not, entern 'None'  for these inputs.
#        x,y are the observations to be differenced; d =y-x. 
#       *Important: If number of success/failures is already known, enter 
#        'None' for these variables 

#       Finally, note that for median comparisons, e.g. is x < mo, y should be 
#       the integer, x the vector.
         
def Bin_test(GLE, n, T, α, x, y ):
    
            
    #========= Set test parameters
    large_sample = 30 # for n > large_sample use z-test.
    greater = GLE[0]  # x > y: if True test p >= po
    lesser  = GLE[1]  # x < y: if True test p <= po
    equals  = GLE[2]  # x = y: if True test p  = po
    
    p  = 0.5  # binomial probability
    
    
    #========= Enter data
    try:
        
        DIM  = x.shape; W = DIM[1]; L = DIM[0]
        n    = W*L
        diff = x-y # take difference from median.
    
    
        #========= find T statistic (count negative vals.)
        # find number of negative difference vals.
        i = 0; T = 0
        while i < L:
            j = 0
            while j < W:
                if diff[i,j] < 0:
                    T += 1
                j += 1
            i += 1
        print(f'T = {T}')
    
    except:
        o = None
    #=================================
    
    if n < large_sample:
        Bt       = BIN_CDF(T,n,p)
        one_Bt_1 = BIN_CDF(T-1,n,p)
        if greater: 
            F = Bt
        if lesser:
            F = one_Bt_1
        if greater or lesser:
            if F < α:
                print(f'F = {np.round(F,4)} < α = {α} : reject.')
            if F >= α:
                print(f'F = {np.round(F,4)} > α = {α}  : cannot reject.')
        if equals:
            if Bt <= α/2 or one_Bt_1 <= α/2:
                print(f'{Bt} < α/2  or  {one_Bt_1} < α/2 : reject.')
            else:
                print(f' {Bt}, {one_Bt_1} > α/2 : accept.')
    
    if n >= large_sample:
        z = (T + 0.5 - n*p) /np.sqrt(n*p*(1-p)) # large sample z-stat
        t = np.sqrt(n*p*(1-p))*z - 0.5 +n*p
        
        if greater or lesser:
            z1_α = Percentile(-5,1-α,Standard_Normal,1)
            t1_α = np.sqrt(n*p*(1-p))*z1_α - 0.5 + n*p
        if lesser:
            if z > z1_α:
                print(f'z = {np.round(z,4)} > z1_α = {np.round(z1_α,4)} : reject.')
            if z < z1_α:
                print(f'z = {np.round(z,4)} < z1_α = {np.round(z1_α,4)} : accept.')
        
        if greater:
            if z < -z1_α:
                print(f'z = {np.round(z,4)} < -z1_α = {np.round(z1_α,4)} : reject.')
            if z > -z1_α:
                print(f'z = {np.round(z,4)} > -z1_α = {np.round(z1_α,4)} : accept.')
        
        return t, t1_α
        
        
        if equals:
            z1_α2 = Percentile(-5,1-α/2,Standard_Normal,1)
            t1_α2 = np.sqrt(n*p*(1-p))*z1_α2 - 0.5 +n*p
            print(f't1_α/2 = {t1_α2}')
            if z < -z1_α2 or z > z1_α2:
                print(f'z = {np.round(z,4)} > z1_α = {np.round(z1_α,4)} \
                      or z = {np.round(z,4)} < -z1_α = {np.round(z1_α,4)}: reject.')
            if z > -z1_α2 and z < z1_α2:
                print(f'z = {np.round(z,4)} < z1_α = {np.round(z1_α,4)} \
                      or z = {np.round(z,4)} > -z1_α = {np.round(z1_α,4)}: accept.')
            
            #==== perform test ========
    
            if z < zα:
                print(f'z = {np.round(z,4)} >= zα = {np.round(zα,4)}')
                print(f'   reject.')
    
            if z >= zα:
                print(f'z = {np.round(z,4)} < zα = {np.round(zα,4)}')
                print(f'   cannot reject.')
            
            return t, t1_α2
            
#===============================
#===============================
# RUN EXAMPLES
print(f'------ \nEXAMPLE 14.5.2')
x  = np.matrix(' 5.09 5.08 5.21 5.07 5.24 5.12 5.16 5.18 2.19 \
                 5.26 5.10 5.28 5.29 5.27 5.09 5.24 5.26 5.17 5.13 \
                 5.27 5.26 5.17 5.19 5.28 5.28 5.18 5.27 5.25 5.26 \
                 5.26 5.18 5.13 5.08 5.25 5.17 5.09 5.16 5.24 5.23 \
                 5.28 5.24 5.23 5.23 5.27 5.22 5.26 5.27 5.24 5.27\
                 5.25 5.28 5.24 5.26 5.24 5.24 5.27 5.26 5.22 5.09')


n = None # number of observations @ < 40,000 - will calculate in code for this example
T = None # number of negative signs - will calculate in code for this example
y = 5.20 ; # median test; y is integer
α = 0.05
GLE = [True, False, False] # test if x0.50 > mo = y = 5.20
Bin_test(GLE, n, T, α, x, y )


#-------------------------------
print(f'\n------ \nEXERCISE 14.11')
x  = np.matrix(' 23 20 26 25 48 26 25 24 15 20')
y  = np.matrix(' 20 30 16 33 23 24  8 21 13 18')
n = None # number of observations @ < 40,000 - will calculate in code for this example
T = None # number of negative signs - will calculate in code for this example  
α = 0.10
GLE = [True, False, False] ; # test if E[x] > E[y]
Bin_test(GLE, n, T, α, x, y )


#---------------------------------
print(f'\n------ \nEXERCISE 14.13.a')
x  = np.matrix(' 111 102 90 110 108 125 99 121 133 115 90 101')
y  = np.matrix(' 97 90 96 95 110 107 85 104 119 98 97 104')
n = None # number of observations @ < 40,000 - will calculate in code for this example
T = None # number of negative signs - will calculate in code for this example  
α = 0.10
GLE = [True, False, False] ; # test if E[x] > E[y]
Bin_test(GLE, n, T, α, x, y )





import sys
sys.exit(0)
 #==========================================
    
# CONFIDENCE INTERVAL (questions 7.b,c)

# see pgs 404-405 and beginning of ch. 14 in ref. 1 of readme file.
# the displayed example is exercies # 7.a in ch. 14.
from helpful_scripts import *
import numpy as np
o  = np.matrix(' 5.09 5.08 5.21 5.07 5.24 5.12 5.16 5.18 2.19 \
                 5.26 5.10 5.28 5.29 5.27 5.09 5.24 5.26 5.17 5.13 \
                 5.27 5.26 5.17 5.19 5.28 5.28 5.18 5.27 5.25 5.26 \
                 5.26 5.18 5.13 5.08 5.25 5.17 5.09 5.16 5.24 5.23 \
                 5.28 5.24 5.23 5.23 5.27 5.22 5.26 5.27 5.24 5.27\
                 5.25 5.28 5.24 5.26 5.24 5.24 5.27 5.26 5.22 5.09')
DIM = x.shape; W = DIM[1]; L = DIM[0]
n = W;   p = 0.25; α= 0.05

#sort observations in ascending order
O = rank_(o)[0] ; #print(f'\nOrdered :\n{O}')


i = 0; sum_ = 1
while sum_ >= 1 - α:
    sum_ += -BIN(i,n,p)
    i += 1
L = i-1
print(F'sum_ = {sum_}')
print(F'L = {L+1}')
print(F'o_Ordered[{L}] = {O[0,L]}')

i = 0; sum_ = 0
while sum_ < 1-α:
    sum_ += BIN(i,n,p)
    i += 1
U = i-1
print(F'sum_ = {sum_}')
print(F'U = {U+1}')
print(F'o_Ordered[{U}] = {O[0,U]}')

#===================================
# for large sample approx. find {zL,zU} then solve for kL,kU.
#zL = Percentile(-5,0.05,Standard_Normal,1)
#print(f'zL = {zL}')
#zU = Percentile(-5,0.95,Standard_Normal,1)
#print(f'zU = {zU}')












