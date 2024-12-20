# SIR colonization model related functions

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint

# The SIR model differential equations.
def SIRderiv(y, t, N, beta, gamma):
    """SIR model derivation.
    
    Args:
        y (tuple): contains the counts for compartments S, I and R.
        t: time point.
        N (int): population size.
        beta (float): transmission coefficient.
        gamma (float): recovery rate.
        
    Returns:
        dSdt: change in S compartment.
        dIdt: change in I compartment.
        dRdt: change in R compartment.
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def SIR(par1, par2, I_scalar, t, N, reparam = False, R_scalar = 0):
    """SIR simulator function. 

    Args:
        par1, par2 (float): parameters of the model. Either the values of beta and gamma, or net transmission or R.
        I_scalar (int): colonization count at time zero.
        t: time grid.
        reparam: if true, the model with net transmission and R is used, and the parameters need to be changed to the beta, gamma scale for the SIR model.
        R_scalar (int): number of recovered at the beginning of the simulation.

    Returns:
        S, I, R (array, int): tuple with three array compartments, containing weekly susceptible, colonized and recovered counts.
    """
    if reparam:
        beta = par1*par2/(par2-1) #par1/(1 - 1/par2) # par1 = net transmission, par2 = R
        gamma = par1/(par2 - 1)
    else:
        beta = par1
        gamma = par2
        
    I0 = np.array([I_scalar]*beta.size)
    R0 = np.array([R_scalar]*beta.size)
    S0 = N - I0 - R0

    # Initial conditions vector
    # Integrate the SIR equations over the time grid, t.
    S = np.zeros((beta.size, t.size))
    I = np.zeros((beta.size, t.size))
    R = np.zeros((beta.size, t.size))
    for i in range(beta.size):
        y0 = S0[i], I0[i], R0[i]
        ret = odeint(SIRderiv, y0, t, args=(N, beta[i], gamma[i]))
        S[i,:] = ret[:,0]
        I[i,:] = ret[:,1]
        R[i,:] = ret[:,2]
    return S, I, R

