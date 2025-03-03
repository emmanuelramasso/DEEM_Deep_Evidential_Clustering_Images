#!/bin/bash
#
# When DML models have been obtained, we get one model per epoch and 
# we want to find the best model. This function loops through multiple 
# repositories, process DML model checkpoint files and append performance 
# metrics to the repository-specific output CSV file (the repo of the DML model)
# The performances are from kmeans on train set + generalisation on the test set
# After that use "tracer_perf.py" to plot the performance and select the best model.
#
# E. Ramasso, FEMTO-ST, helped by ChatGPT, 2024.
#

#ROOT_REPO_DIR="/home/emmanuel.ramasso/Documents/PROGRAMMES/GITHUB/Evidential_neural_network_clustering/MATLAB/test_triplet_constrative/DML_MNIST_prior_margin8/part2"
#ROOT_REPO_DIR="/mnt/tmp_ssd/TESTDEEM/test_triplet_constrative/DML_MNIST_prior_margin8"
ROOT_REPO_DIR="/home/emmanuel.ramasso/Documents/PROGRAMMES/GITHUB/Evidential_neural_network_clustering/MATLAB/test_triplet_constrative/DML_MNIST_prior_margin8/forConstrainedKmeans/"

# Loop through each repository in the root directory
for repo in $ROOT_REPO_DIR/model_DML*/; do
    echo "Processing repository: $repo"
    
    # Extract the value of p from the repo name (e.g., model_DML0.01_repo -> p=0.01)
    repo_name=$(basename $repo)
    p_value=$(echo $repo_name | sed -E 's/model_DML([0-9.]+)_repo/\1/')

    # Define the output file specific to the repository, using p_value in the filename
    OUTPUT_FILE="${repo%/}/performance_metrics_p${p_value}.csv"  # Remove trailing slash from repo
    OUTPUT_DIR="${repo%/}"

    # Header for the output file
    echo "Model,prob,ARItrain,NMItrain,ACCtrain,ARItest,NMItest,ACCtest" > $OUTPUT_FILE

    # Loop through each model file in the repository, only processing 'DML_model_epoch_xxxx.pth' files
    for model in $repo/DML_model_epoch_*.pth; do
        if [[ -f "$model" ]]; then  # Check if the model file exists
            echo "Processing $model with p=$p_value..."
            python Contrastive_DML_eval.py -m $model -o $OUTPUT_DIR -f $OUTPUT_FILE -d 'mnist' -p $p_value
        fi
    done
done
