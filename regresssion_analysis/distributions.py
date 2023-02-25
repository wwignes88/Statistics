

import numpy as np
import math as m


# standard normal pdf.
def Standard_Normal(z, args = None):
    pi    = np.pi
    C     = 1/(np.sqrt(2*pi))
    norm  = C*np.exp(-0.5*(z**2))
    return norm


def student_t(t,*args):
    v = args[0]
    pi = np.pi
    A  = m.gamma((v+1)/2)
    B  = m.gamma((v/2))*np.sqrt(v*pi)
    C  = (1 + t**2/v)**(-(v+1)/2)
    return A*C/B


# x > 0
def Gamma(x,*args):
    #input(args[0])
    θ,k = args[0][0:2]
    A = 1/((θ**k)*m.gamma(k))
    B = (x**(k-1))*np.exp(-x/θ)
    return A*B

def Chi_(x,*args):
    v = args[0]
    θ = 2   ; globals()['θ'] = θ
    k = v/2 ; globals()['k'] = k
    ARGS = θ,k
    return Gamma(x,ARGS)


# x > 0, v >= 0
def Snedecor(x,*args):
    v1,v2 = args[0]
    A = (m.gamma(v1/2)*m.gamma(v2/2))/(m.gamma((v1+v2)/2))
    B = (v1/v2)**(v1/2)
    C = x**(v1/2-1)
    D = (1+x*v1/v2)**(-v1/2-v2/2)
    return B*C*D/A











