# This file calls EVNN.py while setting hyperparameters
# The EVNN.py file generates tensorboard files, with filename suffix
# set in the json file

import json
import subprocess
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
import uuid
from numpy import exp

# function that updates the json file
def update_config(base_config_file, mat_file_name, probprior, nitermax_CyclicLR):
    with open(base_config_file, 'r') as f:
        config = json.load(f)

    if "_campF_captA_" in mat_file_name:
        config["tb_filesuffix"] = "_campF_captA_pp" + str(probprior) + "_"
    if "_campF_captB_" in mat_file_name:
        config["tb_filesuffix"] = "_campF_captB_pp" + str(probprior) + "_"
    if "_campF_captC_" in mat_file_name:
        config["tb_filesuffix"] = "_campF_captC_pp" + str(probprior) + "_"       

    config["mat_file_name"] = mat_file_name
    config["probprior"] = probprior
    config["nitermax_CyclicLR"] = nitermax_CyclicLR
    
    unique_config_file = f'config_{uuid.uuid4()}.json'
    with open(unique_config_file, 'w') as f:
        json.dump(config, f, indent=4)

    return unique_config_file

# make a run
def run_experiment(config_file):
    subprocess.run(['python3', 'EVNN.py', '--jsonfile', config_file])
    os.remove(config_file)  # Clean up the unique config file after the run
    

# For one task
# A=[0	800000
# 0.1	80000
# 0.2	80000
# 0.3	60000
# 0.4	60000
# 0.5	50000
# 0.6	50000
# 0.7	30000
# 0.8	30000
# 0.9	20000
# 1	20000]
# fitobject = fit(A(:,1),A(:,2),'exp2')
# figure,plot(A(:,1),A(:,2),'o'),hold on, plot(A(:,1),fitobject(A(:,1)),'b*')
# 
# fitobject = 
#      General model Exp2:
#      fitobject(x) = a*exp(b*x) + c*exp(d*x)
#      Coefficients (with 95% confidence bounds):
#        a =   7.002e+05  (6.732e+05, 7.272e+05)
#        b =      -266.5  (-1.165e+11, 1.165e+11)
#        c =   9.977e+04  (7.715e+04, 1.224e+05)
#        d =      -1.484  (-1.96, -1.007)

def execute_task(mf, pp, base_config_file, nitermax_CyclicLR_p0, nitermax_CyclicLR_p1):
    a,b,c,d = 7.002e+05, -266.5, 9.977e+04, -1.484
    nit = a*exp(b*pp) + c*exp(d*pp)
    print(f"Nb iter for current p: {nit}")
    unique_config_file = update_config(base_config_file, mat_file_name=mf, probprior=pp, nitermax_CyclicLR=nit)
    run_experiment(unique_config_file)
        
# Main        
def main():
    base_config_file = 'config_exp.json'  # The JSON file to be modified

    mat_file_name = ["DijDij_orionAE_500_campF_captA_sansnormalisation.mat",
                     "DijDij_orionAE_500_campF_captB_sansnormalisation.mat", 
                     "DijDij_orionAE_500_campF_captC_sansnormalisation.mat"]   
    probprior = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]   
    nitermax_CyclicLR_p0 = 800000 # for p=0.0
    nitermax_CyclicLR_p1 = 20000  # for p=1.0

    tasks = list(itertools.product(mat_file_name, probprior))
    num_workers = 4
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(execute_task, mf, pp, base_config_file, nitermax_CyclicLR_p0, nitermax_CyclicLR_p1) for mf, pp in tasks]
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
