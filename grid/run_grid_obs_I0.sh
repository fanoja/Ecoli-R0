#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

conda activate ecoli-elfi

cp grid_params/final/grid_params_A_I0.py grid_params.py
python3 run_grid.py

cp grid_params/final/grid_params_A_no_I0.py grid_params.py
python3 run_grid.py

cp grid_params/final/grid_params_C2_I0.py grid_params.py
python3 run_grid.py

cp grid_params/final/grid_params_C2_no_I0.py grid_params.py
python3 run_grid.py