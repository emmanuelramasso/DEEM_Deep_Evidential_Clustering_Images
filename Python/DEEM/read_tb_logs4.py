# Installation d'un environnement spécifique pour le tracé pour éviter conflits
# source ~/Documents/NONSYNC/plotTensorboardData/bin/activate
# python3 read_tb_logs4.py 
# NOTE : since test is done at the same iteration as validation we look
# for the closest iteration between val->test to get the perf on test after 
# selecting the best val

from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import re

""" TAGS PRESENT IN THE DEEM TENSORBOARD FILE
Losses/Loss_dijkij
Losses/loss_prior
Losses/loss_prior_frDML
Losses/loss_ML_v2
Losses/loss_CNL_v2
Losses/Loss_tot
Visu/loss_ML_v1
Visu/loss_CNL_v1
Visu/plijSij_ML
Visu/plijSijbarre_ML
Visu/plijSij_CNL
Visu/plijSijbarre_CNL
Gradients/totalGrad
Losses/Loss_val
perfVal/aRI
perfVal/aNMI
perfVal/ACC
perfTest/aRI
perfTest/aNMI
perfTest/ACC
LR
Gamma/Gamma
 """

# Root directory containing all folders with TensorBoard logs organized by experiment
# Each subfolder corresponds to a different experiment setup and contains TensorBoard log files
#folder_root = '/home/emmanuel.ramasso/Documents/PROGRAMMES/GITHUB/Evidential_neural_network_clustering/MATLAB/testPython/modelsDMLtestIJAR/modelsDMLpriorlabels/'
#folder_root = '/home/emmanuel.ramasso/Documents/PROGRAMMES/GITHUB/Evidential_neural_network_clustering/MATLAB/testPython/modelsDMLtestIJAR/modelsDMLpriorpairs/'
folder_root = '/home/emmanuel.ramasso/Documents/PROGRAMMES/GITHUB/Evidential_neural_network_clustering/MATLAB/testPython/resnet_lasttest/'

# Specify tags for validation and test metrics
val_tag = 'perfVal/aRI'  # Validation metric
val_tag = 'perfTest/aNMI'  # Validation metric
test_tag = 'perfTest/aNMI'  # Test metric

# Output CSV file
#output_file = 'NMI_performance_DEEM_results_priorlabels.csv'
output_file = 'aRI_performance_DEEM_results_priorlabels.csv'

# Find all event files recursively under folder_root
def find_event_files(directory):
    """
    Recursively finds all TensorBoard event files in the specified directory.

    Args:
        directory (str): The root directory to search for TensorBoard event files.

    Returns:
        list: A list of file paths to TensorBoard event files.
    """
    event_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    return event_files

# Extract prior quantity and type from folder name
def extract_prior_info(folder_name):
    """
    Extracts prior quantity and type based on folder naming conventions.

    Args:
        folder_name (str): The name of the folder containing prior information.

    Returns:
        tuple: A tuple containing the prior quantity (float) and prior type (str).
    """
    match = re.search(r'DML(\d+\.\d+)_test(\d+\.\d+)(priorlabels|priorpair)', folder_name)
    if match:
        prior_quantity = float(match.group(2))  # Extract prior quantity
        prior_type = match.group(3)  # Extract prior type
        return prior_quantity, prior_type
    return None, None

# Process a single event file
def process_event_file(event_file, val_tag, test_tag):
    """
    Processes a TensorBoard event file to retrieve the best validation metric
    and the corresponding test metric.

    Args:
        event_file (str): Path to the TensorBoard event file.
        val_tag (str): Tag for validation metric.
        test_tag (str): Tag for test metric.

    Returns:
        tuple: Best validation metric and corresponding test metric (float, float).
    """
    val_steps, val_values = [], []
    test_steps, test_values = [], []

    for event in summary_iterator(event_file):
        for value in event.summary.value:            
            if value.tag == val_tag:
                val_steps.append(event.step)
                val_values.append(value.simple_value)
                            
            if value.tag == test_tag:
                test_steps.append(event.step)
                test_values.append(value.simple_value)
                
    if not val_values or not test_values:
        return None, None

    # Find best validation metric
    best_val_idx = np.argmax(val_values)
    best_step = val_steps[best_val_idx]
    
    # Find closest test step
    closest_test_idx = np.argmin(np.abs(np.array(test_steps) - best_step))
    best_test_value = test_values[closest_test_idx]

    return val_values[best_val_idx], best_test_value

# Process all directories and output results
results = []
for root, dirs, files in os.walk(folder_root):
    if 'tensorboardlog' in root:  # Check for TensorBoard logs
        print(f"Processing folder: {root}")
        event_files = find_event_files(root)

        # Extract prior information
        folder_name = os.path.basename(os.path.dirname(root))
        prior_quantity, prior_type = extract_prior_info(folder_name)

        for event_file in event_files:
            val_perf, test_perf = process_event_file(event_file, val_tag, test_tag)

            if val_perf is not None and test_perf is not None:
                results.append([folder_name, prior_quantity, prior_type, val_perf, test_perf])
                print(f"Folder: {folder_name}, Prior: {prior_quantity} ({prior_type}), Validation: {val_perf}, Test: {test_perf}")
            else:
                print('Val perf or test perf are empty ?')
                print(val_perf)
                print(test_perf)
                
# Write results to CSV
# Write results to a CSV file, including prior information and performance metrics
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Folder', 'Prior Quantity', 'Prior Type', 'Best Validation', 'Test at Best Validation'])
    writer.writerows(results)

print(f"Results saved to {output_file}")
