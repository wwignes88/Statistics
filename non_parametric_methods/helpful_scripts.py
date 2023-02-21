import numpy as np
from itertools import permutations
import math as m
#==================================
# calculate the inverse of a mtarix
def inverse_(M):
    try:
        return np.linalg.inv(M)
    except:
        input("Singular Matrix, Inverse not possible.")

    
#===================================
# shuffle an input vector (presumed to be in [row] matrix form) in 
# ascending order returns shuffled vector as well as a permutation
# matrix which can act on any other vector of the same 
# length and shuffle it in a way which replicates
# how the input vector has been shuffled.

def shuffle(x):
    x = np.array(x)[0]
    L = len(x)
    I = np.eye(L)
 
    k = 0 
    while k < L-1:
        I_ = np.eye(L)
        x_ = x[k:]
        #print(f'\nx_ = {x_}')
        x_min = np.min(x_)
        #print(f'x_min   = {x_min}')
        min_ind = np.where(x_ == x_min)[0][0]
        #print(f'min_ind = {min_ind}')
        x[[k,min_ind+k]] = x[[min_ind+k, k]]
        I_[:,[k,min_ind+k]] = I_[:,[min_ind+k, k]]
        I = np.matmul(I_,I)
        #input(f'x = {x}')
        k += 1
       
    return np.matrix(x), np.matrix(I)



#===========================================
# sum positive or negative values of a matrix.
    # bin_ is binary variable. 
    # 1 = sum positive values
    # 0 = sum negative values
def pos_neg_sum(M,bin_):

    dim  = M.shape ; W = dim[1] ; L = dim[0]
    sum_ = 0
    i = 0
    while i < L:
        j = 0
        while j < W:
            mij = M[i,j]
            if bin_ == 1:
                if mij > 0:
                    sum_ += mij

            if bin_ == 0:
                if mij < 0:
                    sum_ += mij
                    
            j += 1
            
        i += 1
    return sum_

def index_pos_neg(v):
    i = 0 
    neg_ind = []
    pos_ind = []
    while i < v.shape[1]:
        vi = v[0,i]
        if vi > 0:
            pos_ind.append(i)
        if vi < 0:
            neg_ind.append(i)
        i += 1
    return neg_ind, pos_ind



#==========================================
# assign ranks to [ABSOLUTE] vector v values
# input: v ;presumed single row np.matrix
# also it is presumed that 
def rank_(v):
    vo = np.copy(v)
    # shuffle v in ascending order. 
    # I_ is permutation matrix to replicate this shuffling.
    v, I_ =  shuffle(abs(v)) 
    v = np.matmul(I_,vo.transpose()).transpose()
    
    N = v.shape[1]
    ranks = np.zeros(N) ;  ranks = np.matrix(ranks)
    i = 0 ; 
    while i <  N:  
        ranks[0,i] = i+1
        i += 1
    return v, ranks, I_

#---------------
# same as above but will average equal ranks.
def rank_avg(v):
    
    # shuffle v in ascending order. 
    # I_ is permutation matrix to replicate this shuffling.
    v, I_ =  shuffle(v) 
    
    N = v.shape[1]
    ranks = np.zeros(N) ;  ranks = np.matrix(ranks)
    i = 0 ; 
    while i <  N:
        vi = v[0,i]
        
        # if vi==vj pool ranks, take average:
        if i < N-1 and v[0,i+1] == vi:
            vj = v[0,i+1]
            j  = i+1; sum_ = i+1; count = 1
            while vj == vi:
                sum_ += j+1; count += 1
                j    += 1
                vj = v[0,j]
            rankj = sum_/count 
            ranks[0,i:j] = rankj
            i = j
        else:
            ranks[0,i] = i+1
            i += 1
    return v, ranks, I_


#===========================================
# input: vector (single row of matrix) 
#        [0,0,a,0,b,c]
# i.e. a, b, c are assigned ranks according to the order
# in which they appear.
# output: summation of ranks:
#       Σ[0,0,3,0,5,6] = 12

def rank_sum(v):
    i = 0 ; sum_ = 0
    while i < v.shape[1]:
        sum_ += i+1
        i += 1
    return sum_


# sum rank of ones in binary vector.
def binary_indexed_sum(binaries,x):
    i = 0 ; sum_ = 0
    while i < binaries.shape[1]:
        if binaries[0,i] == 1:
            sum_ += x[0,i]
        i += 1
    return sum_


#==================================
# find critical x value (xc) where cumulative distribution
# equals Y where 0 <= Y <= 1.
def Percentile(x0,Y,pdf,*argv):
    h     = 1/5000
    xi    = x0  ;  SUM   = 0
    while SUM < Y:
        f0   = pdf(xi,*argv)
        fh   = pdf(xi+h,*argv)
        SUM  += fh*h  #0.5*(f0+fh)*h # trapezoidal integration
        xi   += h
    return xi
#========================================
# standard normal pdf.
def Standard_Normal(z, args = None):
    pi    = np.pi
    C     = 1/(np.sqrt(2*pi))
    norm  = C*np.exp(-0.5*(z**2))
    return norm

# CDF of stnadard normal
def Φ(za,zb):
    h  = 1/3000 ; s  = za; SUM = 0 
    while s < zb:
        f0   = Standard_Normal(s)
        fh   = Standard_Normal(s+h)
        SUM  += 0.5*(f0+fh)*h 
        s    += h
    return SUM

#====================================
# Binomial distribution: n >= 0, 0 <= p <= 1 
# also note:error will rise if x input not integer.
def BIN(x,n,p):
    k   = int(x )
    NUM = m.factorial(n) * (p**x) * ((1-p)**(n-x))
    DEN = m.factorial(k) * m.factorial(n-k)
    return NUM/DEN

def BIN_CDF(x,n,p):
    s = 0; sum_ = 0
    while s <= x:
        sum_ += BIN(s,n,p)
        s += 1
    return sum_
    



#====================================
# input  : row vector (presumed to be in [row] matrix form)
# returns: matrix whose rows are all UNIQUE permutations of the input.
def permute_list(v):
    v = np.array(v)[0,:] # presumed matrix row vector
    L = len(v)
    
    b  = list(permutations(v))
    Lb = len(b)
    
    p = []       # permutation vectors
    no_list = [] # index repeat permutation vectors
    
    i = 0 ;
    while i < Lb:
        bi = b[i] 
        j = i+1
        while j < Lb:
            if bi == b[j]:
                no_list.append(j)
            j += 1
        if i not in no_list:
            p.append(bi)
        i +=  1
    
    p = np.matrix(p)
    return p   

#==========================================

# input  : single row of matrix (i.e. a vector in matrix form)
# returns: matrix whose rows are all UNIQUE permutations of the input.
def eliminate_repeat_rows(M):
    L = len(M)
    i = L-1 ;
    while i > 0:
        Mi = M[i]
        
        j  = i-1
        while j >= 0:
            if np.all(Mi == M[j]):
                M  = np.delete(M,j,axis = 0)
                i +=-1
            j += -1
 
        i +=  -1

    return M
