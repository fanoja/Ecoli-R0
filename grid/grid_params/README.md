# Settings for grid-based simulation

## Instructions

- Copy and paste the contents of a relevant directory (for example, `104_weeks`) to this directory. 
- Run `run_all_grids.sh`
- Enjoy!

## Contents

In each directory there are different sets of grid parameters to run the following combinations:
- Synthetic data, not aggregated (specify true beta/gamma parameters here)
- Synthetic data, aggregated
- Synthetic data, reparametrized (net transmission & R), not aggregated
- Synthetic data, reparametrized, aggregated
- Observed data
- Observed data, reparametrized

- `104_weeks`: Run the simulations above for 104 weeks
- `full weeks`: Run the simulations for 16*52 weeks (the entire duration of the study)