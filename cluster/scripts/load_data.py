import os, re
import pandas as pd

cwd = os.getcwd()

if bool(re.search('cluster', cwd)):
    cwd = cwd[0:len(cwd) - len("/cluster")] # uncomment if operating in the cluster/ directory
print(f"Current working directory: {cwd}")


# load babybiome data (OR data)
or_data = pd.read_excel(f"{cwd}/data/ST131_clades_OR_E_coli_carriage_disease_collapsed_wBSAC2.xlsx")
print(or_data.head())


def get_obs_BSI(df, clade, cladecol = 'clade', is_prop = True):
    # Get the proportion of a clade out of all ST131 observations per year.
    # Deprecated.
    
    import pandas as pd
    
    if 'clade' in df.columns:
        cladecol = 'clade'
        
    if is_prop:
        theta_BSI_obs = pd.value_counts(df.loc[df[cladecol] == clade]["year"])/pd.value_counts(df["year"])# n clades per year/n all ST131 obs
    else:
        theta_BSI_obs = pd.value_counts(df.loc[df[cladecol] == clade]["year"]).sort_index() # these are counts directly

    
    return theta_BSI_obs.fillna(0) # assume that years with missing obs did not have any BSI cases.

def get_incidence_data(csv_file, clade = "A", is_prop = True, n_incidence_pop = 1000000, partial_time = False):
    # Get the BSI clade X incidence per 1000000 people.
    # If is_prop = True, divides the incidence by n_incidence_pop
    
    import pandas as pd

    csv_file = 'data/NORM_incidence.csv'
    df = pd.read_csv(csv_file, delimiter=',')
    
    rnames = df["Year"]
    
    df = df[clade]
    
    if is_prop:
        df = df/n_incidence_pop
     
    df.index = rnames
    
    if partial_time:
        start_year_i = 6
        end_year_i = 12
        
        df = df.iloc[start_year_i:end_year_i]
        
    return df
