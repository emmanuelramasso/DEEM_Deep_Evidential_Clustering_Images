# this file is made to load a distance metric learning model
# to be used in EVNN
# we load first
# we apply the model on an image 
# it generates the embedding

from EVNN_config import device, args
import torch
import torchvision.transforms as transforms
import json 
    
# Load the pre-trained model
def load_pretrained_DML():
    # load model 
    n = args['model_path_DML']+args['model_name_DML']
    model_metric = torch.jit.load(n+'.pth').to(device)
    model_metric.eval()
    
    # load input shape    
    with open(n+'_input_shape.json', 'r') as f:
        input_shape_dict = json.load(f)
    input_shape = input_shape_dict['input_shape']    
    print(f'The shape of images before DML will be reshaped into {input_shape} to fit DML model')
    
    # load mean and std
    #running_mean = torch.load(n+'_running_mean.pt')
    #running_std = torch.load(n+'_running_stddev.pt')
    running_mean = torch.load(n+'_batch_mean.pt')
    running_std = torch.load(n+'_batch_stddev.pt')
    print(f'running mean train {running_mean}')
    print(f'running std train {running_std}')
    transform_DML = transforms.Compose([
        transforms.Resize((input_shape[1],input_shape[2])),        
        ])
    normalisation_embedding = {'mean': torch.tensor(running_mean).to(device), 'std': torch.tensor(running_std).to(device)}
    
    # load percentiles to compute delta_0    
    with open(n+'_percentiles_dict.json', 'r') as f:
        percentiles_dict = json.load(f)    
    print(f'The percentiles dictionary is {percentiles_dict}')
    
    # load the indices of labeled data used in DML
    # labeled_indices = torch.load(n+'_labeled_indices.pt')
    # DML uses "pairs" of similar or dissimilar or unsupervised objects
    # instead of loading indices for which we now the label, we here load pairs
    # of indices with associated boolean 1 or 0
    indicesAPN_all = torch.load(n+'_indicesOfPairs.pt')
    bools_APN_all = torch.load(n+'_boolSimilarOfPairs.pt')

    return model_metric, transform_DML, normalisation_embedding, percentiles_dict, indicesAPN_all, bools_APN_all
