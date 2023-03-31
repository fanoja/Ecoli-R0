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
        
        
        
## SIR for elfi

# Simulator with counts

def SIR_simulator(beta, gamma, nt = 104, N = 100000, batch_size = 1, random_state = None):
    
    print(f'beta:{beta}, gamma:{gamma}')
    #params = {"S0": 100000 - 1, "I0": 1, "R0":0, "times":[i for i in range(0,104)],"timestep":1, "N":100000}
    S = np.zeros((batch_size, nt))
    I = np.zeros((batch_size, nt))
    R = np.zeros((batch_size, nt))
    
    # set initial values
    S[:,0] = N - 1
    I[:,0] = 1
    R[:,0] = 0
    
    N = np.array([N]*nt)
    
    for t in range(0,nt-1):
        
        if t == 0:
            print(S[:,t])
        
        S_next = S[:,t] + dS(S, I, t, beta, N, is_prop = False)
        if S_next[0] < 0:
            pass
            #print("Negative S value!")
            #print(f'Beta:{beta}, gamma: {gamma}')
            
        S[:,t + 1] = S_next
        
        I_next = I[:,t] + dI(I, S, t, beta, gamma, N, is_prop = False)
        if I_next[0] < 0:
            pass
            #print("Negative I value!")
            #print(f'Beta:{beta}, gamma: {gamma}')
            
        I[:,t + 1] = I_next
        R[:,t + 1] = R[:,t] + dR(I, t, gamma)
    
    return S, I, R
    
    
def col_to_BSI_count(SIR, OR_hat, N = 100000, theta_C = 1, theta_BSI = 0.3):
    
    N_CA = SIR[1] # batch_size x n_obs
    
    #N = max(SIR[0][0] + 1) # assumes that I0 = 1 and S0 = N - I0.

    N_C = theta_C*N
    
    if N_C < np.max(N_CA, axis = 1)[0]: # TODO checks only the first batch here.
        print("Warning! N_C < N_CA!!")
        print("Tune the starting parameters.")
        
    if N_C < 0:
        print("Warning! N_C < 0")
        
    
    bs = N_CA.shape[0] # batch size. dim(SIR) = (bs, T)
    T = N_CA.shape[1]
    N_C0 = N_C - N_CA
    # choose the clade and dataset
    #df = or_data[or_data["Label"] == f'{clade} ({dataset})']
    
    # odds ratios from a normal distribution for a bit of uncertainty (based on data)
    #or_mu = df["OR"]
    #or_sd = (df["upper"] - df["lower"])/2
    
    #OR_hat = np.random.normal(or_mu, or_sd**2, 1)
    #OR_hat = np.array([np.random.normal(or_mu, or_sd**2, 1)]*bs) # here just one OR taken for a given simulation.
    
    N_BSI = theta_BSI*N#[int(df["Disease_PP"]) + int(df["Disease_nonPP"])]*T
    
    N_BSI_A = OR_hat.reshape(-1,1)*N_CA*N_BSI/(OR_hat.reshape(-1,1)*N_CA + N_C0)
    
    return N_BSI_A


def plot_col_to_BSI_count(SIR, or_data = or_data, clade = "A", dataset = "NORM", n_rep = 100):
    # Plot n_rep repetitions of theta_BSI_clade as "translated" from colonization by clade of interest.
    
    
    all_bsi_reps = []
    for i in range(0, n_rep):
        
        OR_hat = get_OR_hat(or_data = or_data, clade = clade, dataset = dataset)
        obsBSI = col_to_BSI_count(SIR, OR_hat)
        if i == 0:
            plt.plot(obsBSI[0], color = "lightblue", label = "N_BSI_A")
        else:
            plt.plot(obsBSI[0], color = "lightblue")
        all_bsi_reps.append(obsBSI)

    plt.plot(SIR[1][0], color = "red", label = "N_C_A")
    plt.plot(np.mean(all_bsi_reps, axis = 0)[0], color = "navy", label = "Mean of BSI reps")
    plt.title(f"N BSI: Clade {clade}, {dataset}")
    plt.xlabel("Years")
    plt.ylabel("N")
    plt.legend()
    plt.show()
