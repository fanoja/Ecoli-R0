#!/bin/bash -l
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

conda activate ecoli-elfi

# Run all clade A alpha grids

cp grid_params/alpha_test/grid_params_A_alpha_0.py grid_params.py
python3 run_grid.py


cp grid_params/alpha_test/grid_params_A_alpha_1.py grid_params.py
python3 run_grid.py


cp grid_params/alpha_test/grid_params_A_alpha_2.py
python3 run_grid.py


cp grid_params/alpha_test/grid_params_A_alpha_3.py grid_params.py
python3 run_grid.py


cp grid_params/alpha_test/grid_params_A_alpha_4.py grid_params.py
python3 run_grid.py


cp grid_params/alpha_test/grid_params_A_alpha_5.py grid_params.py
python3 run_grid.py


cp grid_params/alpha_test/grid_params_A_alpha_6.py grid_params.py
python3 run_grid.py


cp grid_params/alpha_test/grid_params_A_alpha_7.py grid_params.py
python3 run_grid.py


cp grid_params/alpha_test/grid_params_A_alpha_8.py grid_params.py
python3 run_grid.py
