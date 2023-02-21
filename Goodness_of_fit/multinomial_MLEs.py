import math as m
import numpy as np
import warnings

# Some required distributions:
#----------------------------
# x > 0
def Gamma(x,*args):
    #input(args[0])
    θ,k = args[0][0:2]
    A = 1/((θ**k)*m.gamma(k))
    B = (x**(k-1))*np.exp(-x/θ)
    return A*B

def Chi(x,*args):
    v = args[0]
    θ = 2   ; globals()['θ'] = θ
    k = v/2 ; globals()['k'] = k
    ARGS = θ,k
    return Gamma(x,ARGS)

#--------------------
# standard normal pdf.
def Standard_Normal(z, args = None):
    pi    = np.pi
    C     = 1/(np.sqrt(2*pi))
    norm  = C*np.exp(-0.5*(z**2))
    return norm

# CDF of stnadard normal
def Φ(za,zb):
    h  = 1/10000 ; s  = za; SUM = 0 
    while s < zb:
        f0   = Standard_Normal(s)
        fh   = Standard_Normal(s+h)
        SUM  += 0.5*(f0+fh)*h 
        s    += h
    return SUM

#---------------------
# see pg. 456 of reference 1 in readme file.
# This is the MLE equation for σ
def fσ( μ, σ, *args):
    O, intervals = args[0][0:2]
    
    i = 0; sum_ = 0
    while i < len(O):
        intvli = intervals[i]
        
        oi   = O[i]
        ai   = intvli[0]
        ai_1 = intvli[1]
        
        zi   = (ai   - μ)/σ
        zi_1 = (ai_1 - μ)/σ
        
        #------------------------
        # integral standard normal variables
        Φi_prime   = Standard_Normal(zi)
        Φi_1_prime = Standard_Normal(zi_1)
        
        NUMi  = zi_1*Φi_1_prime - zi*Φi_prime 
        DENi  = Φ(zi,zi_1)
        if DENi == 0:
            print(f'    ****skipped {i}******')
            #warnings.warn("terms have been ommitted due to divide by zero.")
            
        if DENi > 0:
            sum_ += oi*NUMi/DENi
        i += 1
    print(f'    σ sum_ = {sum_}')
    return sum_



#---------------------
# see pg. 456 of reference 1 in readme file.
# This is the MLE equation for μ
def fμ( μ, σ, *args):
    O, intervals = args[0][0:2]
    
    i = 0; sum_ = 0
    while i < len(O):
        intvli = intervals[i]
        
        oi   = O[i]
        ai   = intvli[0]
        ai_1 = intvli[1]
        
        #  print(f'\na{i} = {ai}')
        #  print(f'μ = {μ}')
        #  input(f'σ = {σ}')
        
        zi   = (ai   - μ)/σ
        zi_1 = (ai_1 - μ)/σ
        
        #------------------------
        # integral standard normal variables
        Φi_prime   = Standard_Normal(zi)
        Φi_1_prime = Standard_Normal(zi_1)
        
        NUMi  = Φi_1_prime - Φi_prime 
        DENi  = Φ(zi,zi_1)
        if DENi == 0:
            print(f'    ****skipped {i}******')
            #warnings.warn("terms have been ommitted due to divide by zero.")
            
        if np.all(DENi > 0):
            sum_ += oi*NUMi/DENi
        i += 1
    print(f'    μ sum_ = {sum_}')
    return sum_


#----------------------------------------



