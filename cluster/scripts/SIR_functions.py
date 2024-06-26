# SIR model related functions

import numpy as np
    
def dS(S, I, t, beta, N, is_prop = False):
    #print(f"dS is_prop: {is_prop}")
    if is_prop:
        #print(f'S[t]: {S[:,t].shape}')
        return -beta*S[:,t]*I[:,t]#.reshape(-1,1)
    return -beta*S[:,t]*I[:,t]/N[t]

def dS_SIS(S, I, t, beta, gamma, N, is_prop = False):
    if is_prop:
        return -beta*S[:,t]*I[:,t] + gamma*I[:,t]
    return -beta*S[:,t]*I[:,t]/N[t] + gamma*I[:,t]

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
    
def check_SIR_nonneg(comp_t, dcomp, is_prop = False):
    # Checks that the new value in this compartment is nonnegative. If not, add zero to comp
    # Check also that no compartment goes over 1
    # Note: This is for proportional SIR!
    # comp: compartment of interest, S, I or R for example
    # dcomp: change in the compartment
    # comp_t: current value in the compartment
    
    comp_t1 = comp_t + dcomp
    
    comp_t1[comp_t1 < 0] = 0 # set negative values to zero
    
    if is_prop:
        comp_t1[comp_t1 > 1] = comp_t[np.where(comp_t1 > 1)] # If any proportion goes above 1 after addition
    else:
        comp_t1[comp_t1 < 1] = 0 # set values less than 1 to zero
    return comp_t1


def SIS(par1, par2, nt, N, I0 = None, reparam = False, is_prop = True, batch_size = 1, random_state = None):
    
        #print(f"is_prop in SIR: {is_prop}")
    par1 = np.atleast_1d(par1)
    par2 = np.atleast_1d(par2)
    batch_size = par1.shape[0]
    thetaS = np.zeros((batch_size, nt))
    thetaI = np.zeros((batch_size, nt))
    
    if I0 == None:
        thetaS[:,0] = N-1
        thetaI[:,0] = 1
        #print(thetaI[:,0])
    else:
        thetaI[:,0] = I0
        thetaS[:,0] = N - I0
    
    if is_prop:
        thetaS[:,0] = thetaS[:,0]/N # recommendation: make S0 the same as N - I0
        thetaI[:,0] = thetaI[:,0]/N
    
    N = np.array([N]*nt)
    
    if reparam: # Lintusaari et al 2016
        
        # If there has been a log transformation of the parameters. If not, comment out.
        #par1 = np.exp(par1)
        #par2 = np.exp(par2)
        
        a = par1*par2/(par2-1) #par1/(1 - 1/par2) # par1 = net transmission, par2 = R
        b = par1/(par2 - 1)
    else:
        a = par1
        b = par2

    
    for t in range(0, nt-1):

        thetaS[:,t + 1] = check_SIR_nonneg(thetaS[:,t], dS_SIS(thetaS, thetaI, t, a, b, N, is_prop = is_prop), is_prop = is_prop)
        thetaI[:,t + 1] = check_SIR_nonneg(thetaI[:,t], dI(thetaI, thetaS, t, a, b, N, is_prop = is_prop), is_prop = is_prop)
        
    return thetaS, thetaI

def SIR(par1, par2, nt, N, I0 = None, reparam = False, is_prop = True, batch_size=1, random_state = None):
    
    #print(f"is_prop in SIR: {is_prop}")
    par1 = np.atleast_1d(par1)
    par2 = np.atleast_1d(par2)
    batch_size = par1.shape[0]
    thetaS = np.zeros((batch_size, nt))
    thetaI = np.zeros((batch_size, nt))
    thetaR = np.zeros((batch_size, nt))
    
    if I0 == None:
        thetaS[:,0] = N-1
        thetaI[:,0] = 1
        #print(thetaI[:,0])
    else:
        thetaI[:,0] = I0
        thetaS[:,0] = N - I0
    
    thetaR[:,0] = 0
    
    if is_prop:
        thetaS[:,0] = thetaS[:,0]/N # recommendation: make S0 the same as N - I0
        thetaI[:,0] = thetaI[:,0]/N
        thetaR[:,0] = thetaR[:,0]/N
    
    N = np.array([N]*nt)
    
    if reparam: # Lintusaari et al 2016
        
        # If there has been a log transformation of the parameters. If not, comment out.
        #par1 = np.exp(par1)
        #par2 = np.exp(par2)
        
        a = par1*par2/(par2-1) #par1/(1 - 1/par2) # par1 = net transmission, par2 = R
        b = par1/(par2 - 1)
    else:
        a = par1
        b = par2

    
    for t in range(0, nt-1):

        thetaS[:,t + 1] = check_SIR_nonneg(thetaS[:,t], dS(thetaS, thetaI, t, a, N, is_prop = is_prop), is_prop = is_prop)
        thetaI[:,t + 1] = check_SIR_nonneg(thetaI[:,t], dI(thetaI, thetaS, t, a, b, N, is_prop = is_prop), is_prop = is_prop)
        thetaR[:,t + 1] = check_SIR_nonneg(thetaR[:,t], dR(thetaI, t, b), is_prop = is_prop)
        
    return thetaS, thetaI, thetaR


def SIR_tau(par1, par2, nt, N, I0 = None, reparam = False, is_prop = True, batch_size=1, random_state = None):
    
    #print(f"is_prop in SIR: {is_prop}")
    par1 = np.atleast_1d(par1)
    par2 = np.atleast_1d(par2)
    batch_size = par1.shape[0]
    thetaS = np.zeros((batch_size, nt))
    thetaI = np.zeros((batch_size, nt))
    thetaR = np.zeros((batch_size, nt))
    
    par2 = 1/par2 # Using mean infection time as par2 instead of infection rate
    
    if I0 == None:
        thetaS[:,0] = N-1
        thetaI[:,0] = 1
        #print(thetaI[:,0])
    else:
        thetaI[:,0] = I0
        thetaS[:,0] = N - I0
    
    thetaR[:,0] = 0
    
    if is_prop:
        thetaS[:,0] = thetaS[:,0]/N # recommendation: make S0 the same as N - I0
        thetaI[:,0] = thetaI[:,0]/N
        thetaR[:,0] = thetaR[:,0]/N
    
    N = np.array([N]*nt)
    
    if reparam: # Lintusaari et al 2016
        
        # If there has been a log transformation of the parameters. If not, comment out.
        #par1 = np.exp(par1)
        #par2 = np.exp(par2)
        
        a = par1*par2/(par2-1) #par1/(1 - 1/par2) # par1 = net transmission, par2 = R
        b = par1/(par2 - 1)
    else:
        a = par1
        b = par2

    
    for t in range(0, nt-1):

        thetaS[:,t + 1] = check_SIR_nonneg(thetaS[:,t], dS(thetaS, thetaI, t, a, N, is_prop = is_prop), is_prop = is_prop)
        thetaI[:,t + 1] = check_SIR_nonneg(thetaI[:,t], dI(thetaI, thetaS, t, a, b, N, is_prop = is_prop), is_prop = is_prop)
        thetaR[:,t + 1] = check_SIR_nonneg(thetaR[:,t], dR(thetaI, t, b), is_prop = is_prop)
        
    return thetaS, thetaI, thetaR


def prop_to_nSIR(SIRsim, N):
    # Convert proportions to counts in a SIR model
    
    S = SIRsim[0]
    I = SIRsim[1]
    R = SIRsim[2]
    
    return S[:,]*N, I[:,]*N, R[:,]*N



# ELFI related functions

# Distance metrics

def I_mean(y):
    #print(y)
    #print(y[1])
    return np.mean(y[1][:,], axis=1)

def I_var(y):
    return np.var(y[1][:,], axis=1)

def I_max(y):
    return np.max(y[1][:,], axis = 1)

def S_min(y):
    return np.min(y[0][:,], axis=1)

def R_max(y):
    return np.max(y[2][:,], axis=1)

def I_max_bsi(y):
    I_max = np.max(y[:,], axis = 1)
    return I_max#.reshape(-1,1)

def I_mean_bsi(y):
    I_mean = np.mean(y[:,], axis = 1)
    return I_mean#.reshape(-1,1)

def I_var_bsi(y):
    return np.var(y[:,], axis = 1)#.reshape(-1,1)