import matplotlib.pyplot as plt
import numpy as np
from multinomial_MLEs import *

#----------------------------
# find critical x value (xc) where cumulative distribution
# equals Y where 0 <= Y <= 1.
def Percentile(x0,Y,pdf,*argv):
    h     = 1/55000
    xi    = x0  ;  SUM   = 0
    while SUM < Y:
        f0   = pdf(xi,*argv)
        fh   = pdf(xi+h,*argv)
        SUM  += fh*h  #0.5*(f0+fh)*h # trapezoidal integration
        xi   += h
    return xi

#----------------------------


def PLOT_MLEs(N_grid,grid_view , *args): 
    ARGS = args[0]
    # importing modules
    from mpl_toolkits import mplot3d
    import numpy
    from matplotlib import pyplot
    
    
    # Create meshgrid: σ is x-axis. μ = y-axis
    N = 2 # number of grid points
    f_mat = np.zeros([N+1,N+1])
    zero_plane = np.zeros([N+1,N+1])
    
    # create initial/ final vals, step sizes.
    μ  = -30;  μf = -5 ; hμ  = (μf - μ)/N 
    σ  = 10 ;  σf = 25 ; hσ  = (σf - σ)/N
    
    # create meshgrid plotting arrays.
    μ_arr = np.array([])
    σ_arr = np.array([])
    k = 0 
    while k <= N:
        σ_arr  = np.append(σ_arr,σ)
        μ_arr  = np.append(μ_arr,μ)
        σ += hσ
        μ += hμ
        k += 1
        
    # z = f(x,y) grid
    i = 0
    while i <= N:
        j = 0
        μ = μ_arr[i]
        while j <= N:
            σ = σ_arr[j]
            f_mat[i,j] = fμ( μ, σ, ARGS)
            #print(f'finished j = {j}')
            j += 1
        #print(f'finished i = {i}')
        i += 1
    
    x, y = np.meshgrid(σ_arr, μ_arr)
    z    = f_mat
    
    
    # creating the visualization
    fig = pyplot.figure()
    wf = pyplot.axes(projection ='3d')
    #wf.plot_wireframe(x, y, z, color ='green')
    
    surf = wf.plot_surface(x, y, z, cmap = plt.cm.cividis)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    surf = wf.plot_surface(x, y, zero_plane, alpha = 0.7, color = 'red')
    
    
    #------------------------
    plt.xlabel('σ')
    plt.ylabel('μ')
    
    Xview,Yview = grid_view 
    wf.view_init(Xview, Yview)
    
    # displaying the visualization
    wf.set_title('μ MLE function')
    pyplot.show()







