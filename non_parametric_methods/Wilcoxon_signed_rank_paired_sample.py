# c.f. example 14.5.2
# here we apply the one sample signed rank test to testing 
#        Ho: x_0.50 = y_0.50  ,  x_0.50 > y_0.50 
# where x+0.50 represents the median of X data.

from helpful_scripts import *
import numpy as np



# performs Wilcoxon signed rank test for paired samples.
# c.f. example 14.5.2 in ref. 1 of readme file. 
# for the moment [until I functioanlize the code with a t-table] the user
# neds to look up the appropriate critical value for t (tα) in a table or 
# calculate it.
def Wilcoxon_signed_paired(α,tα,x,y):
    n = x.shape[1]
    print(f'x = \n{x}')
    print(f'y = \n{y}')
    #======== Sort data/ assign ranks =========
    
    d  = x-y # take difference
    print(f'd = \n{d}')
    
    # find index of negative/ positive difference vals.
    neg_ind, pos_ind = index_pos_neg(d)
    
    # sort vals in [absolute] ascending order, assign ranks [ranks not averaged].
    d, d_ranks, I_ = rank_(d) ;
    
    # restore d_ranks according to the original ordering of x,y.
    d_ranks = np.matmul(inverse_(I_),d_ranks.transpose()).transpose()
    print(f'd_ranks = \n{d_ranks}')
    
    
    # Calculated T stats ; *pos_neg_sum addes positive OR negative values 
    T_neg = np.sum(d_ranks[0,neg_ind]) ; print(f'\nT negative = {T_neg}')
    T_pos = np.sum(d_ranks[0,pos_ind]) ; print(f'T positive = {T_pos}')
    t = T_neg
    
    
    # Large sample approximation 
    if n > 30:
        t = np.min([T_neg, T_pos])
        z = (t +.05 - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24) 
        
        # Find critical t value:
        # Large sample critical value:
        zα = Percentile(-10,α,Standard_Normal,1)
        tα  = zα*np.sqrt(n*(n+1)*(2*n+1)/24) + n*(n+1)/4 
        print(f'\n*large sample approximation:')
        print(f'   z  = {np.round(z,4)}:')
        print(f'   zα = {np.round(zα,4)}:')
    
    
    # perform test :
    if t < tα:
        print(f'\nt = : {t} < tα = {tα}')
        print(f'   reject Ho')
    
    if t >= tα:
        print(f'\nt = : {t} >= tα = {tα}')
        print(f'   cannot reject Ho')


# set test level
α  = 0.10   
tα = 60.9   # for small samples (<30) must look up in table.. 
            # for the moment.
    
    
# Enter data:

#print(f'\n------ \nEXAMPLE 14.5.2')
x = np.array([10.8,12.7,13.9,18.1,19.4,21.3,23.5,24.0,24.6,25.0,\
              25.4,27.7,30.1,30.6,32.3,33.3,34.7,38.8,40.3,55.5])
y = np.array([9.8,13.0,10.7,19.2,18.0,20.1,20.0,21.2,21.3,25.5,\
              25.7,26.4,24.5,27.5,25.0,28.0,37.4,43.8,35.8,60.9])

x = np.matrix(x)  
y = np.matrix(y)  ; n = x.shape[1]
α  = 0.05   
tα = 60.9   # for small samples (<30) must look up in table.. 
            # for the moment
Wilcoxon_signed_paired(α,tα,x,y)


#---------------------------------
print(f'\n------ \nEXERCISE 14.13.b')
x  = np.matrix(' 111 102 90 110 108 125 99 121 133 115 90 101')
y  = np.matrix(' 97 90 96 95 110 107 85 104 119 98 97 104')

α  = 0.10   
tα = 60.9   # for small samples (<30) must look up in table.. 
            # for the moment
#Wilcoxon_signed_paired(α,tα,x,y)



