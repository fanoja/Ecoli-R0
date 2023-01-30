### Implementations of the SIR model
# - SIR with counts
# - SIR with proportions

import matplotlib.pyplot as plt


def dS(S, I, t, beta, N):

    return -beta*S[t]*I[t]/N[t]

def dI(I, S, t,beta, alpha, N):
    
    return beta*S[t]*I[t]/N[t] - alpha*I[t]

def dR(I, t, alpha):
    
    return alpha*I[t]


def SIR(beta, alpha, I0 = 1, R0 = 0, S0 = 9999, N = 10000, T = 100):
    # alpha: recovery rate
    # beta: infection rate
    
    S = [S0]
    I = [I0]
    R = [R0]
    N = [N]*(T - 1) #*(len(T)-1) # Assume a fixed population size over time.
    
    for t in range(0, T - 1):
        
        # Update SIR model
        I.append(I[t] + dI(I, S, t, beta, alpha, N))
        R.append(R[t] + dR(I, t, alpha))
        S.append(S[t] + dS(S, I, t, beta, N))
        
    return S, I, R
 

def plotSIR(SIR):
    
    T = len(SIR[0])
    
    plt.plot(T, SIR[0])
    plt.plot(T, SIR[1])
    plt.plot(T, SIR[2])
    plt.title("SIR simulation")
    plt.legend(["S", "I", "R"])
    plt.show()
    
    
def least_squares_estimator(SIR_obs, N, T = 100, plot_loss = False):
    # Simple least squares estimator for calculating beta of a SIR model from data.
    # Assume same initial values as data
    
    I0 = np.ceil(min(SIR_obs[1]))
    R0 = np.ceil(min(SIR_obs[2]))
    S0 = np.ceil(max(SIR_obs[0]))
    
    print("Initial values:", I0, R0, S0)

    T = [i for i in range(0,T)]
    
    S_obs = SIR_obs[0]
    I_obs = SIR_obs[1]
    R_obs = SIR_obs[2]


    alpha = 0.2

    losses = []
    betas = []

    p_test = 0.4 # How much of the observed/simulated distribution is used to calculate the loss
    n_test = int(len(I_obs)*p_test)

    for b in np.arange(0, 2, 0.01): # Iterate over a range of potential values of beta

        new_SIR = SIR(I0, R0, S0, N, T, beta = b, alpha = alpha) # Simulated dataset at the value of beta

        # Compute loss on the number of infected patient
        loss_I = np.sum((np.array(new_SIR[1][0:n_test]) - np.array(I_obs[0:n_test]))**2)
        #loss_S = np.sum((np.array(new_SIR[0]) - np.array(S_obs))**2)
        losses.append(loss_I)
        betas.append(b)


    print("Smallest loss:", np.min(losses))
    print("Achieved at beta =", betas[losses.index(np.min(losses))])
    
    if plot_loss:
        plt.axis([0, max(betas), 0, np.median(losses)])
        plt.plot(betas, losses)
        plt.title("SSE loss vs beta")
        plt.xlabel("beta")
        plt.ylabel("SSE")
        plt.show()