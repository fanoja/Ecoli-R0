# Reparametrized SIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma
import elfi
import scipy.stats


## Simulation options ##

use_obs_data = False # use observed or simulated data
is_agg = True # aggregate data?
clade = "A"
obs_data = "NORM" # NORM, BSAC

theta_bsi = 0.3 # proportion of population interest (age group) with BSI
theta_c = 1 # proportion of colonized

# parameters for simulated data

net_transmission_param = 2
R_param = 5

beta = 0.234
gamma = 0.81

# priors for parameters

param1_prior = elfi.Prior(scipy.stats.uniform,0.01,20)
param2_prior = elfi.Prior(scipy.stats.uniform, 2, 10)



## Loading the data ##

# load NORM data

norm_data = pd.read_excel("data/mmc2.xlsx", engine = 'openpyxl') # this is the NORM data


# load BSAC data
bsac_data = pd.read_csv("data/Supplemental_Data_S1.csv")


df = norm_data


# load babybiome data
or_data = pd.read_csv("data/ST131_clades_OR_E_coli_carriage_disease_collapsed.csv")


# load population data

norway_pop_data = pd.read_csv("data/Norway_population_2002-2017.csv", sep = '\t', header = 0, index_col = 0)


# load age distribution data
ages = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99", "100-109"]
groups = [61, 22, 67, 97, 131, 255, 556, 767, 900, 316, 3]

norm_age_data = pd.DataFrame(data={"age":ages, "n_BSI":groups})

bsac_data = bsac_data.rename(columns = {'Year_of_isolation':'year', 'MLST':'ST', 'Phylogroup':'clade'})
norm_data = norm_data.rename(columns = {'CC131_clades':'clade'})


def get_obs_BSI(df, clade, cladecol = 'clade', is_prop = True):
    # Get the proportion of clade out of all observations per year
    
    if 'clade' in df.columns:
        cladecol = 'clade'
        
    if is_prop:
        theta_BSI_obs = pd.value_counts(df.loc[df[cladecol] == clade]["year"])/pd.value_counts(df["year"])# n clades per year/n all ST131 obs
    else:
        theta_BSI_obs = pd.value_counts(df.loc[df[cladecol] == clade]["year"]).sort_index() # these are counts directly

    
    return theta_BSI_obs.fillna(0) # assume that years with missing obs did not have any BSI cases.

## Observational model ##

def get_OR_hat_pars(or_data, clade = "A", dataset = "NORM"):
    
    df = or_data[or_data["Label"] == f'{clade} ({dataset})']
    or_mu = df["OR"]
    or_sd = (df["upper"] - df["lower"])/2
    
    return or_mu, or_sd
    
def get_OR_hat(or_data, clade = "A", dataset = "NORM", batch_size = 1, random_state = None):
    # TODO: fix that random state
    or_mu, or_sd = get_OR_hat_pars(or_data, clade = clade, dataset = dataset)

    OR_hats = np.empty(batch_size)
    
    for b in range(0, batch_size):
        OR_hat = np.random.normal(or_mu, or_sd**2, 1)

        max_iter = 1000
        i = 0

        while OR_hat[0] < 0:
            OR_hat = np.random.normal(or_mu, or_sd**2, 1)
            i = i + 1
            if i == max_iter:
                break

        if i > 0:
            print(f"Iterated OR_hat {i} times due to negativity.")

        if OR_hat < 0:
            print(f"Warning, negative OR_hat after max iterations!")

        OR_hats[b] = OR_hat
    
    return OR_hats

def col_to_BSI(SIR, OR_hat, theta_c = 1, theta_bsi = 0.3, is_prop = True):
    # SIR: output of the SIR simulator (clade of interest colonization proportion over time)
    # theta_bsi: The proportion of bsi in the entire (colonized) population - from the age distribution.
    # theta_c: The overall proportion of population colonized by E. coli. For simplicity, assume we are only interested in the colonized 
    # population and set theta_c = 1 by default.
    # By changing is_prop = False, can work with counts instead of proportions
    
    theta_c_a = SIR[1]
    
    if not is_prop: 
        N = SIR[0][0][0] + 1 # the first entry in S compartment + 1 is the population size.
        theta_c = N*theta_c 
        theta_bsi = theta_bsi*N
    else:
        if np.min(np.max(SIR[1][:,], axis = 1)) > 1:
            print("Warning! col_to_BSI uses proportions, but SIR seems to use counts.")
    
    bs = theta_c_a.shape[0]
    n_obs = theta_c_a.shape[1] # with batches. 0 = first batch

    theta_c_0 = theta_c - theta_c_a

    theta_bsi_a_hat = OR_hat.reshape(-1,1)*theta_c_a*theta_bsi/(theta_c_0 + OR_hat.reshape(-1,1)*theta_c_a)
    
    return theta_bsi_a_hat


# SIR in terms of t and R
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

def SIR_reparam(net_transmission, R, nt, N, batch_size=1, random_state = None):

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
    
    a = net_transmission/(1 - 1/R)
    b = net_transmission/(R - 1)

    
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

def sum_over_bsi(bsi_obs, time_period = 52):
    # Take a sum over every ith week in bsi_obs (from i to i + time_period, where i is the current week)
    # Note: probably not applicable for proportion observations, only counts
    # time_period: time period to sum over. By default 52 (weeks).
    
    n_years = len(bsi_obs[0])//52
    #print(f'Number of years to sum: {n_years}')
    agg_bsi = []
    for i in range(0, n_years):
        start = i*time_period # which week to start summing at
        end = time_period*i + time_period # next week
        bsi_obs_yearly = bsi_obs[:,start:end]
        agg_bsi.append(np.sum(bsi_obs_yearly[:,], axis = 1))
    agg_bsi = np.asarray(agg_bsi).transpose()   
    
    return agg_bsi

def SIR_and_BSI_simulator(net_transmission, R, nt, N, bsi_pars, is_prop = False, is_agg = False, time_period = 52, batch_size = 1, random_state = None):
    # A simulator function combining both the SIR simulation and the observational model
    
    
    # SIR simulator:
    SIR = SIR_reparam(net_transmission = net_transmission, R = R, nt = nt, N = N, batch_size = batch_size, random_state = random_state)
    
    if not is_prop:
        SIR = prop_to_nSIR(SIR, N)
        
    # Observational model:
    
    or_data = bsi_pars["or_data"]
    clade = bsi_pars["clade"]
    dataset = bsi_pars["dataset"]
    theta_c = bsi_pars["theta_c"]
    theta_bsi = bsi_pars["theta_bsi"]
    
    
    or_hat = get_OR_hat(or_data, clade, dataset, batch_size = batch_size, random_state = random_state)
    
    BSI = col_to_BSI(SIR, or_hat, theta_c = theta_c, theta_bsi = theta_bsi, is_prop = is_prop)
    
    if is_agg:
        BSI = sum_over_bsi(BSI, time_period = time_period)

    return BSI


## ELFI ###

def BSI_max_t(y):
    # time to peak/maximum number of bsi cases
    # shaped (batch_size, n_obs)
    
    return np.argmax(y, axis = 1)

def BSI_max(y):
    # maximum number of BSI cases
    
    max_bsi = np.max(y[:,], axis = 1)
    
    return max_bsi


if obs_data == "NORM":
    bsi_obs_data = get_obs_BSI(df = norm_data, clade = clade, is_prop = True)
else:
    bsi_obs_data = get_obs_BSI(df = bsac_data, clade = clade, is_prop = True)
print(bsi_obs_data)

n_years = max(bsi_obs_data.index) - min(bsi_obs_data.index) # years of interest
n_weeks = (n_years + 1)*52 # weeks of interest
print(f'Number of weeks to simulate: {n_weeks}')

pop_size = 100000
print(f'Population size: {pop_size}')


# A simple SIR with elfi

elfi.new_model()

# simulated data, proportions:
#SIR_obs = propSIR_simulator(0.734, 1/30, nt = n_weeks, N = pop_size, batch_size = 1)   
#OR_hat = get_OR_hat(or_data, clade = "A", dataset = "NORM")
#bsi_obs = col_to_BSI(SIR_obs, OR_hat = OR_hat)

#print(bsi_obs)
#bsi_obs = (aggregate_BSI(bsi_obs, nan_locations = []), aggregate_BSI(bsi_obs, nan_locations = []) , aggregate_BSI(bsi_obs, nan_locations = [])) # no missing values in simulated data
#print(bsi_obs)
#nan_locations = np.where(np.isnan(bsi_obs))[0]
#print(nan_locations)



# simulated data, counts with a summation aggregate:
is_p = True

# Actual data:
#bsi_obs = np.asarray(get_obs_BSI(norm_data, clade = clade)).reshape(1,-1)
#print(bsi_obs)
bsi_pars = {"or_data": or_data, "clade": clade, "dataset": obs_data, "theta_c":theta_c, "theta_bsi":theta_bsi}

if use_obs_data:
    bsi_obs = np.asarray(get_obs_BSI(norm_data, clade = clade)).reshape(1,-1)
else:
    bsi_obs = SIR_and_BSI_simulator(net_transmission_param, R_param, nt = n_weeks, N = pop_size, bsi_pars = bsi_pars, is_prop = is_p, is_agg = agg_bsi, time_period = 52, batch_size = 1, random_state = None)#.flatten()

#bsi_obs = (aggregate_BSI(bsi_obs, nan_locations), aggregate_BSI(bsi_obs, nan_locations), aggregate_BSI(bsi_obs, nan_locations))
print(bsi_obs.shape)
print(bsi_obs.ndim)


#beta = elfi.Prior(scipy.stats.uniform, 0, 1)
#gamma = elfi.Prior(scipy.stats.uniform, 0, 1)
#gamma = elfi.Constant(1/30)

# Clancy et al: uninformative gamma priors for beta and gamma
# Lintusaari et al 2016

net_transmission = param1_prior
#beta = elfi.Prior(scipy.stats.uniform, 0, 1)
R = param2_prior
#gamma = elfi.Prior(scipy.stats.norm,1/30,0.01)
#gamma = elfi.Prior(scipy.stats.uniform, 0, 1)

nt = elfi.Constant(n_weeks)
N = elfi.Constant(pop_size)
is_prop = elfi.Constant(is_p)
is_agg = elfi.Constant(agg_bsi)
time_period = elfi.Constant(52)

bsi_pars = elfi.Constant(bsi_pars)
SIR = elfi.Simulator(SIR_and_BSI_simulator, net_transmission, R, nt, N, bsi_pars, is_prop, is_agg, time_period, observed = bsi_obs)

S1 = elfi.Summary(BSI_max_t, SIR)
S2 = elfi.Summary(BSI_max, SIR)
#S3 = elfi.Summary(I_var_bsi, SIR)

d = elfi.Distance('euclidean', S1, S2)

elfi.draw(d)

elfi.set_client('multiprocessing') # parallellization