# To print out the acceptance rate and threshold values, run this script.
import numpy as np
import pandas as pd

def get_optimal_ABCSMC_AR_summary(res_dir, fname, clade, model):
    # Get acceptance rate vs threshold data, if locally optimal ABC-SMC was used (different output data format)


    result = np.load(res_dir + fname, allow_pickle = True)[()]
    n_pops = len(result["populations"])

    pop_df = pd.DataFrame({"clade": np.repeat(clade, n_pops), "model": np.repeat(model, n_pops), "AR":result["acceptance_rate"], "threshold":result["thresholds"]})

    print(result["acceptance_rate"])
    
    return pop_df

def get_AR_summary(res_dir, clade = None, model = None):
    # Get acceptance rate vs threshold data frame for given clade and model.
    
    test = np.load(res_dir + "result.npy", allow_pickle = True)[()]
    #print(test.summary(all = True))
    
    n_pops = len(test.populations)

    ar_list = []
    for p in range(0, n_pops):
        pop = test.populations[p]
        ar = pop.accept_rate
        n_samples = pop.n_samples
        n_sims = pop.n_sim
        eps = pop.threshold

        #temp_df = pd.DataFrame({"clade":clade, "model":model})
        temp_df = pd.DataFrame({"clade": [clade], "model": [model], "AR":[float(ar)], "threshold":[eps], "n_sims":[n_sims], "n_samples":[n_samples]})
        ar_list.append(float(ar))
        if p == 0:
            pop_df = temp_df
        else:
            pop_df = pd.concat([pop_df, temp_df])

    print(ar_list)
    return(pop_df)




# Clade A

print(get_optimal_ABCSMC_AR_summary("res/final_res/A_results/model_comparison/SIR/result_A_SIR/",\
                fname = "result_A_SIR.npy",\
               clade = "A",\
               model = "SIR"))

print(get_optimal_ABCSMC_AR_summary("res/final_res/A_results/model_comparison/SIS/",\
                fname = "result_A_SIS.npy",\
               clade = "A",\
               model = "SIS"))


# Clade C2
print(get_AR_summary("res/final_res/C2_results/model_comparison/SIR/clade_C2_2025-07-02_14-01-49_10k_h_eps/",\
               clade = "C2",\
               model = "SIR"))

print(get_optimal_ABCSMC_AR_summary("res/final_res/C2_results/model_comparison/SIS/result_C2_SIS/",\
                fname = "result_C2_SIS.npy",\
               clade = "C2",\
               model = "SIS"))


# Clade C1

#get_AR_summary()

print(get_AR_summary("res/final_res/C1_results/model_comparison/SIR/clade_C1_2025-07-03_08-31-29_SIR_h_eps/",\
               clade = "C1",\
               model = "SIR"))

print(get_AR_summary("res/final_res/C1_results/model_comparison/SIS/clade_C1_2025-07-02_19-22-52/",\
               clade = "C1",\
               model = "SIS"))


