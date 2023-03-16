# SIR model related functions

import numpy as np
    
def dS(S, I, t, beta, N, is_prop = False):
    if is_prop:
        #print(f'S[t]: {S[:,t].shape}')
        return -beta*S[:,t]*I[:,t]#.reshape(-1,1)
    return -beta*S[:,t]*I[:,t]/N[t]

def dI(I, S, t,beta, gamma, N, is_prop = False):
    
    if is_prop:
        return beta*S[:,t]*I[:,t] - gamma*I[:,t]
    return beta*S[:,t]*I[:,t]/N[t] - gamma*I[:,t]

def dR(I, t, gamma):
    
    return gamma*I[:,t]

def plot_SIR(SIR):
    # plot SIR
    import matplotlib.pyplot as plt
    
    # plot first batch of S, I and R
    plt.plot(SIR[0][0], label = "S") 
    plt.plot(SIR[1][0], label = "I")
    plt.plot(SIR[2][0], label = "R")
    plt.title("SIR")
    plt.legend(loc = "upper right")
    
    plt.show()
    
def check_SIR_nonneg(comp_t, dcomp):
    # Checks that the new value in this compartment is nonnegative. If not, add zero to comp
    # Check also that no compartment goes over 1
    # Note: This is for proportional SIR!
    # comp: compartment of interest, S, I or R for example
    # dcomp: change in the compartment
    # comp_t: current value in the compartment
    
    comp_t1 = comp_t + dcomp
    
    comp_t1[comp_t1 < 0] = 0 # set negative values to zero
    comp_t1[comp_t1 > 1] = comp_t[np.where(comp_t1 > 1)] # If any proportion goes above 1 after addition
    
    return comp_t1

def SIR(par1, par2, nt, N, reparam = False, batch_size=1, random_state = None):
    
    thetaS = np.zeros((batch_size, nt))
    thetaI = np.zeros((batch_size, nt))
    thetaR = np.zeros((batch_size, nt))
    
    thetaS[:,0] = N-1
    thetaI[:,0] = 1
    thetaR[:,0] = 0
    
    thetaS[:,0] = thetaS[:,0]/N # recommendation: make S0 the same as N - I0
    thetaI[:,0] = thetaI[:,0]/N
    thetaR[:,0] = thetaR[:,0]/N
    
    N = np.array([N]*nt)
    
    if reparam:
        a = par1/(1 - 1/par2)
        b = par1/(par2 - 1)
    else:
        a = par1
        b = par2

    
    for t in range(0, nt-1):

        thetaS[:,t + 1] = check_SIR_nonneg(thetaS[:,t], dS(thetaS, thetaI, t, a, N, is_prop = True))
        thetaI[:,t + 1] = check_SIR_nonneg(thetaI[:,t], dI(thetaI, thetaS, t, a, b, N, is_prop = True))
        thetaR[:,t + 1] = check_SIR_nonneg(thetaR[:,t], dR(thetaI, t, b))
        
    return thetaS, thetaI, thetaR


def prop_to_nSIR(SIR, N):
    # Convert proportions to counts in a SIR model
    
    S = SIR[0]
    I = SIR[1]
    R = SIR[2]
    
    return S[:,]*N, I[:,]*N, R[:,]*N