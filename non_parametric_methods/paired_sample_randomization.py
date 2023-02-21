

from helpful_scripts import *
from itertools import permutations
import numpy as np


# Test for randomness between two vectors x,y. If the difference 
# between two vectors is a random quantity we'd expect the median
# of the difference d between them to be md=0. Hence our test
#              Ho: md = 0
# Let T be the sum of positive difference values. 

# Actually no test is performed as the user must decide what constitutes a 
# small value of T. So rather than having a critical value as we often do for 
# comparison, here we instead develop a critical region α based on the observed 
# value of T itself. 

# The value ofα is α=k/N where k,N are to be explained now.

# There being  N = 2**|len(x)| ways to assign + or - vals to |len(x)| set of numbers,
# our critical value is α=k/N val where k is the number of these possible permutations  
# which can be formed such that the sum of positive values are less than 
# the observed value T, e.g. here we have T=7, so we find all possible
# permutations of the difference vector d such that the sum of positive 
# values is less than or equal to 7. Call the resulting set of vstacked 
# vectors our randomiztion table (printed out). The length of this table 
# constitutes the number of permutations which sum to 7 or less. 

# *Warning!: if the sum is over seven, and if some of the difference values happen 
# to be small in absolute magnitude, then it is possible that a permutation of seven 
# different numbers may be required. It is impracticle to permute 7! combinations.
# integer optimization techniques can allow us to restrict the permuations,
# but this is an interesting problem that I must table for the moment.

def paired_sample_randomization(x,y):
    d = y-x
    L = d.shape[1]
    
    # perform 
    
    #============ calculat N, k vals ============
    N = 2**L # number of distinct ways to assign +,- vals to L numbers.
    
    # add positive values of o (call it T). We we will then permute all possible
    # positive combinations in o which are greater or less than T
    T = pos_neg_sum(d, 1) 
    
    #=========================================
    # make all values negative
    d = -abs(d) ; d_copy = np.copy(d)
    rand_table = d # initialize randomization table
    
    # permute different combinations of +/- signs for difference (di) values
    # with the criteria that the sum of positive values is less than 
    # 5+2 = 7 = T 
        
    i = L - 1;  
    while i > L-T:
        permuted_vec = d[0,i:] ; Wp = permuted_vec.shape[1]
        #print(f'n===========================')
        #print(f'permuted_vec {i} : {permuted_vec}')
        ones = np.matrix(np.ones(Wp))
        j = 0 
        while j < Wp:
            ones[0,j] = -1
            p = permute_list(ones) ; Lp = len(p)
            #print(f'\n-----------\np ')
            
            permuted = np.multiply(p,permuted_vec)
            #print(f'permuted: \n{permuted}')
            if j == 0:
                l = p
            m = 0
            while m < len(permuted):
                table_row    = d_copy
                d_copy[0,i:] = permuted[m]
                if pos_neg_sum(table_row,1) <= T:
                    rand_table = np.vstack([ rand_table, table_row ])
                    #print(f'table_row: {table_row}')
                m += 1
            j += 1
        i += -1
        
    a  =  rand_table
    rand_table = eliminate_repeat_rows(rand_table)
    print(f'rand_table: \n{rand_table}')
    
    
    
    #========== compare to critical value ============
    k = len(rand_table) 
    α = k / N  
    
    print(f'\nT = {T}') # critical value.
    print(f'α = {k}/{N} = {α} confidence.') # critical value.


# See page 482 of ref. 1 in readme.
d = np.matrix('-20 -10 -8 -7 5 -4 2 -1')
# given d, we want x,y such that d = y-x.
y = np.matrix(' 0 0 0 0 0 0 0 0')
x = -np.matrix('-20 -10 -8 -7 5 -4 2 -1')


paired_sample_randomization(x,y)














