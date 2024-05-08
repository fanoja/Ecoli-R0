# Estimating R0 for Ecoli

Code for the estimation of R0 for *E. coli* colonisation.
- Model: SIR + observational model
- Method: likelihood-free inference

## Structure

- `doc/`: environment.yml file can be found here
- `data/`: datasets
- `cluster/scripts/`: Files for 
- The main notebook is `Model 1.ipynb`

### Other relevant notebooks

- `Priors.ipynb`: Identifiability and testing different priors.
- `summaries.ipynb`: Testing summaries for ELFI.
- `diagnostics.ipynb`: Validation and diagnostics related code.
- `exploratory.ipynb`: Exploratory figures (replicates of previous studies and some original figures also).
- `SIR models.ipynb`: Notebook with a simple SIR model and least squares estimation (for practice). 
- `reparam SIR.ipynb`: Reparametrized model with net transmission and R as the parameters to estimate.


### Identifiability


## Identifiable
Simulated data, not aggregated:
- `beta = elfi.Prior(scipy.stats.gamma, 1, 0, 10)` and `gamma = elfi.Prior(scipy.stats.norm,1/30,0.01)`
- `beta = elfi.Prior(scipy.stats.gamma, 1, 0, 100)` and `gamma = elfi.Prior(scipy.stats.norm,1/30,0.01)`
- `beta = elfi.Prior(scipy.stats.gamma, 1, 0, 1000)` and `gamma = elfi.Prior(scipy.stats.norm,1/30,0.01)`

## Not identifiable


## Slow:
- Aggregated data