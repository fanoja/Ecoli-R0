import numpy as np
import matplotlib.pyplot as plt

def plot_priors_elfi():
    # Plot priors from the elfi model. Currently supports only (beta, gamma) parametrisation.
    
    prior_sample =  m.generate(1000, outputs = ["gamma", "beta"])
    g = prior_sample["gamma"]
    b = prior_sample["beta"]
    a_sample = alpha.generate(1000)


    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0, 0].hist(b, bins=20)
    axs[0, 0].set_title('Beta prior')
    axs[0, 1].hist(g, bins=20)
    axs[0, 1].set_title('Gamma prior')

    axs[1, 0].scatter(b,g, s = 2)
    axs[1, 0].set_title('Joint prior')
    axs[1, 0].set_xlabel('Beta')
    axs[1, 0].set_ylabel('Gamma')

    axs[1,1].hist(a_sample, bins=20)
    axs[1, 1].set_title('Alpha prior')
    plt.tight_layout()
    plt.show()


    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    nt_sample = b - g
    R_sample = b/g
    axs[0].hist(nt_sample)
    axs[0].set_title("Corresponding net transmission")

    axs[1].hist(R_sample)
    axs[1].set_title("Corresponding R")
    plt.show()


    plt.scatter(nt_sample, R_sample, s=2)
    plt.title("Corresponding net transmission and R joint prior")
    plt.show()
    


    