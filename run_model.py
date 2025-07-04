# Run the model
import importlib
import sys

# Import the ELFI model
import elfi_functions
importlib.reload(elfi_functions)
from elfi_functions import get_model, run_model

## Run the model ##

model_config_file = sys.argv[1]
model = get_model(model_config_file)
run_model(model, model_config_file)



    

    

