
import numpy as np
from multinomial_MLEs import *



# -----------------------------------
# Raphson 2D
# assumed that f, g accept inputs in the form
#            f(xn,yn, args)

def newton_raphson_2D(f , g,  x0, y0, *args):
    
    args = args[0]
    
    h    = 1/2000
    Tol_ = 0.001
    n = 0; kill_ = False
    while n < 1000:
        print(f'\n--------Raphson {n}:')
   
        # find derivatives of f, g  @ x0, y0:
        f0 = f(x0, y0, args) 
        fx = (f(x0 + h, y0 , args) - f0)/h
        fy = (f(x0 , y0 + h, args) - f0)/h
        
        g0 = g(x0, y0, args) 
        gx = (g(x0 + h, y0 , args) - g0)/h
        gy = (g(x0 , y0 + h, args) - g0)/h

        # perform Newton-Raphson iteration
        J  = fx*gy - fy*gx # equivalent to |Jacobian|
        xn = x0 + (-f0*gy + fy*g0)/J
        yn = y0 + (-fx*g0 + f0*gx)/J
        
        fn = f(xn,yn, args) ; gn = g(xn,yn, args)

        if abs(fn) < Tol_ and abs(gn < Tol_):
            print(f'\n\n Quite @ n = {n}')
            print(f'f(n) = {fn}')
            print(f'g(n) = {gn}')
            print(f'xn, yn = {xn,yn}')
            return xn, yn
        
        if n == 100:
            raise ValueError("Newton-Raphson is not converging.")

        # create plot vectors
        print(f'x(n) = {x0} ; y(n) = {y0}')
        print(f'f(n) = {f0} ; g(n) = {g0}')
        
        x0 = xn ; y0 = yn
        n += 1


# -----------------------------------
# should be updated with *args for f(x,args)
# but the 1D newton-raphson function is not used here.
# Raphson 1D
def newton_raphson(f, f0 , μ0, σ0 ):
    x_plot = np.array([])
    f_plot = np.array([])
    
    h    = 1/10000
    Tol_ = 0.00001
    n = 0; kill_ = False
    while n < 1000:

        # find derivative of f @ x0:
        f0 = f(x0)
        print(f'f(n) = {f0}')
        print(f'x(n) = {x0}')
        fh = f(x0 + h) 
        f_derivative = (fh - f0)/h 
        
        # perform Newton-Raphson iteration
        xn = x0 - f(x0)/f_derivative
        fn = f(xn,args)
        
        # create plot vectors
        x_plot = np.append(x_plot, x0)
        f_plot = np.append(f_plot, f0)  

        if n == 100:
            raise ValueError("Newton-Raphson is not converging.")
            
        if fn < Tol_:
            
            x_plot = np.append(x_plot, xn)
            f_plot = np.append(f_plot, f(xn))
            print(f'\n Quite @ n = {n}')
            print(f'f(n) = {fn}')
            print(f'x(n) = {x0}')
            plt.plot(x_plot,f_plot)
            return xn

        x0 = xn
        n += 1












