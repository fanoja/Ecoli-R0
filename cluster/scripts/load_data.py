import os, re
import pandas as pd

cwd = os.getcwd()

if bool(re.search('cluster', cwd)):
    cwd = cwd[0:len(cwd) - len("/cluster")] # uncomment if operating in the cluster/ directory
print(f"Current working directory: {cwd}")

# load NORM data

norm_data = pd.read_excel(f"{cwd}/data/mmc2.xlsx", engine = 'openpyxl') # this is the NORM data


# load BSAC data
bsac_data = pd.read_csv(f"{cwd}/data/Supplemental_Data_S1.csv")


df = norm_data


# load babybiome data
or_data = pd.read_csv(f"{cwd}/data/ST131_clades_OR_E_coli_carriage_disease_collapsed.csv")


# load population data

norway_pop_data = pd.read_csv(f"{cwd}/data/Norway_population_2002-2017.csv", sep = '\t', header = 0, index_col = 0)


# load age distribution data
ages = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99", "100-109"]
groups = [61, 22, 67, 97, 131, 255, 556, 767, 900, 316, 3]

norm_age_data = pd.DataFrame(data={"age":ages, "n_BSI":groups})

bsac_data = bsac_data.rename(columns = {'Year_of_isolation':'year', 'MLST':'ST', 'Phylogroup':'clade'})
norm_data = norm_data.rename(columns = {'CC131_clades':'clade'})

def get_obs_BSI(df, clade, cladecol = 'clade', is_prop = True):
    # Get the proportion of clade out of all observations per year
    
    import pandas as pd
    
    if 'clade' in df.columns:
        cladecol = 'clade'
        
    if is_prop:
        theta_BSI_obs = pd.value_counts(df.loc[df[cladecol] == clade]["year"])/pd.value_counts(df["year"])# n clades per year/n all ST131 obs
    else:
        theta_BSI_obs = pd.value_counts(df.loc[df[cladecol] == clade]["year"]).sort_index() # these are counts directly

    
    return theta_BSI_obs.fillna(0) # assume that years with missing obs did not have any BSI cases.
