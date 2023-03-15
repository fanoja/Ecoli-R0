# SIR model (proportions)

    
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
    

def propSIR_simulator(beta, gamma, nt, N, is_bsi = False, bsi_pars = None, agg_bsi = False, is_prop = True, nan_locations = [], batch_size = 1, random_state = None):
    # SIR model with proportions
    
    
    #bsi_params = {"theta_C":0.6, "theta_BSI": 0.05, "mu_OR":0.4, "var_OR":0.6, "OR_size":100}
    #params = {"S0": 100000 - 1, "I0": 1, "R0":0}
    #print(f'beta: {beta}, gamma: {gamma}')
    
    #beta = np.asanyarray(beta).reshape((-1, 1))
    #gamma = np.asanyarray(gamma).reshape((-1, 1))
    
    #print(f'beta shape: {beta.shape}, gamma shape: {gamma.shape}')
    
    import numpy as np
    thetaS = np.zeros((batch_size, nt))
    thetaI = np.zeros((batch_size, nt))
    thetaR = np.zeros((batch_size, nt))
    
    thetaS[:,0] = N-1
    thetaI[:,0] = 1
    thetaR[:,0] = 0
    
    if is_prop:
        thetaS[:,0] = thetaS[:,0]/N # recommendation: make S0 the same as N - I0
        thetaI[:,0] = thetaI[:,0]/N
        thetaR[:,0] = thetaR[:,0]/N
    
    N = np.array([N]*nt)

    for t in range(0, nt-1):

        thetaS[:,t + 1] = thetaS[:,t] + dS(thetaS, thetaI, t, beta, N, is_prop = is_prop)
        thetaI[:,t + 1] = thetaI[:,t] + dI(thetaI, thetaS, t, beta, gamma, N, is_prop = is_prop)
        thetaR[:,t + 1] = thetaR[:,t] + dR(thetaI, t, gamma)

    if is_bsi:
        
        or_data = bsi_pars["or_data"]
        clade = bsi_pars["clade"]
        dataset = bsi_pars["dataset"]
        theta_c = bsi_pars["theta_c"]
        theta_bsi = bsi_pars["theta_bsi"]
        
        
        or_hat = get_OR_hat(or_data = or_data, clade = clade, dataset = dataset, batch_size = batch_size, random_state = random_state)
        bsi = col_to_BSI((thetaS, thetaI, thetaR), OR_hat, theta_c = 1, theta_bsi = 0.3)
        #print(len(bsi))
        
        if agg_bsi:
            bsi = aggregate_BSI(bsi, nan_locations = nan_locations, batch_size = batch_size)
            return bsi, bsi, bsi
        
        return bsi
        
    return thetaS, thetaI, thetaR