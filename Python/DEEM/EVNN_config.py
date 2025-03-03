import torch
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import os 

########################################################
# Device configuration, GPU or CPU
def device_config():    
    use_cuda = torch.cuda.is_available()    
    if use_cuda:
        device = torch.device("cuda")
        idcuda = 0
        print(f"Device name: {torch.cuda.get_device_name(idcuda)}\n")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print('NO GPU !')       
    print(device)
    return device

device = device_config() 

########################################################
# json file loader for hyperparam and dataset
def load_hyperparameters_from_file(file_path):
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)
        return hyperparameters
    
# Load the json file
parser = argparse.ArgumentParser()
parser.add_argument('-json', '--jsonfile', required=True, type=str, help='Path to json config filename')
argsJson = vars(parser.parse_args())
config_file = argsJson['jsonfile']
args = load_hyperparameters_from_file(config_file)    
print(args)

# Create a repo with model files1
dest_dir = ''
if args['training_mode'] == "DML":
    dest_dir = args['model_name_DML'] + '_'
dest_dir += args['tb_filesuffix'] + '_DEEM_repo'
os.makedirs(dest_dir, exist_ok=False)
tbdir = dest_dir+'/'+'tensorboardlog'
os.makedirs(tbdir, exist_ok=False)
    
# Tensor board
log_dir = tbdir #"logs/"  #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir,filename_suffix=args['tb_filesuffix'])
matplotlib.use('Qt5Agg')

# dataset
dataset = args['dataset']
NUMCLASSES = args['numclasses']
#nbFocsets = args['nbfocsets']
SIZE_IMAGES = tuple(args['input_shape'])
network = args['network']

if dataset == 'mnist':
    coloredinput=False 
elif dataset == 'orion':
    coloredinput=True
        