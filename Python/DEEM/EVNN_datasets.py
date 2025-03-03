# python3 EVNN.py --jsonfile config000001.json 

from __future__ import print_function
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import json 

from EVNN_config import NUMCLASSES, args, device

import os
import random
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from EVNN_DML import load_pretrained_DML
from EVNN_opt import sanity_check_1cycle
from collections import defaultdict
from EVNN_utils import compute_90th_percentile_matrix

##############################################################################
###### METHOD 1
######
# We generated the dij in advance (using matlab codes)
# We load the .mat with images, labels, dij and indices
######
###### METHOD 2
######
# We trained a DML model to generate embeddings, like using triplet ANP
# We load the model here and apply on a image
# Using it on a pair of images i j, we can then generate dij
# Note that the distance should be the same as in DML training
######
##############################################################################



##############################################################################
##############################################################################
###### METHOD 2
##############################################################################
##############################################################################

##############################################################################
# TO LOAD THE DATA
##############################################################################

def data_loader_DML(dataset_name):
    
    # load the model for DML
    try: 
        model_metric, transform_DML, normalisation_embedding, percentiles_dict, indicesAPN, boolsAPN = load_pretrained_DML()
    except Exception as e:
        model_metric, transform_DML, normalisation_embedding, percentiles_dict, indicesAPN, boolsAPN = None, None, None, None, None, None
        print('NO DML MODEL')            
        
    images_path = args['image_folder_train']

    # We need to ensure consistent image size with DML
    image_shape = args['input_shape']
    transform = transforms.Compose([
        transforms.Resize((image_shape[1],image_shape[2])),
        transforms.ToTensor()
        ])
    
    # now load the data, class number first
    print(f'Load dataset {dataset_name}')
    print(f'from {images_path}')
    # get the image folder names first, to get the corresponding labels without specific mapper -- we use the one on imagefolder
    tmp = datasets.ImageFolder(root=images_path)
    # Print out the classes and their corresponding indices
    print("Classes found in training: ", tmp.classes)
    if len(tmp.classes) != NUMCLASSES:
        raise ValueError("Number of classes provided in json is incorrect")
    
    print("Class to index mapping: ", tmp.class_to_idx)
    class_names_to_idx = tmp.class_to_idx       
        
    # create dataset
    # do we need to use the indices of labeled data ??
    # No use of prior on pairs from DML
    if args['useDMLprior_pairs']==0 and args['useDMLprior_pairs2labels']==0:
        indicesAPN = None
        boolsAPN = None
        
    # load the precomputed if needed
    indexingEmbeddings = None
    if args['training_mode'] == "PRECOMPUTED_EMBEDDINGS":
        with open(args['jsonfile_precompEmbeddings'], "r") as file:
            indexingEmbeddings = json.load(file)
        # compute an estimate of 95th percentile of distances
        embeddings = torch.tensor(list(indexingEmbeddings.values()), device=device)  # Shape: (n_samples, n_features)
        d090 = compute_90th_percentile_matrix(embeddings, sample_size=1000) 
        print(f'Initial estmate of 90th percentiles of distances is {d090}')         
    else: 
        # computed in DML 
        d090 = percentiles_dict['prct90th_alldist']
        
    print('Create the training dataset...')        
    train_dataset = SemiSupervisedImagePairDataset(
                        root_dir=images_path,  
                        class_names_to_idx=class_names_to_idx, 
                        model_DML=model_metric, 
                        transform=transform, 
                        transform_DML=transform_DML,
                        probabilityUsePrior=args['probprior'], 
                        indicesAPN=indicesAPN, 
                        boolsAPN=boolsAPN,
                        prob_sampleforceDMLprior_pairs=args['prob_sampleforceDMLprior_pairs'], 
                        prob_sampleforce_DEEM_pairs=args['prob_sampleforce_DEEM_pairs'],
                        indexingEmbeddings=indexingEmbeddings
                        )

    # PAS DE VALIDATION OU FAIRE UN JEU A PART COMME LE TEST !!!
    # Split the dataset into training and validation  --- pb de correspondance avec DML...
    #print('Split into train/validation datasets...')
    #train_dataset, val_dataset = splitCustom(train_dataset, args['train_val_split'] )
    print('taille des données TRAIN : ',len(train_dataset))    
    
    # better to keep shuffle false for consistency with DML files order
    #train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)            
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=True)            
    
    if args['scheduler'] == 'OneCycleLR':
        sanity_check_1cycle(len(train_dataset))
               
    print('Loading test data...')
    if dataset_name=="mnist":
        transformTest = transforms.Compose([
            transforms.Grayscale(), # because image loader loads RGB! 
            transforms.ToTensor(), 
            ])
    elif dataset_name=="orion":
        transformTest = transform
        
    test_dataset = datasets.ImageFolder(root=args['image_folder_test'], transform=transformTest)
    #test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True, drop_last=False)            
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, pin_memory=True, drop_last=False)            
    print('Taille des données TEST : ',len(test_dataset))
    
    if args['image_folder_valid'] != "" and args['training_mode'] == "PRECOMPUTED_EMBEDDINGS":
        raise "Error: Implementation not ready to use precomputed embedding inthe validation stage, use empty validation folder"
    
    if args['image_folder_valid'] != "":
        val_dataset = SemiSupervisedImagePairDataset(
                        root_dir=args['image_folder_valid'],  
                        class_names_to_idx=class_names_to_idx, 
                        model_DML=model_metric, 
                        transform=transform, 
                        transform_DML=transform_DML,
                        probabilityUsePrior=0.0, 
                        indicesAPN=None, 
                        boolsAPN=None,
                        prob_sampleforceDMLprior_pairs=0.0
                        )
        #val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)            
        val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=True)            
        print('Taille des données VAL : ',len(val_loader.dataset))
    else:
        val_loader = []
        print('No validation')
    
    return train_loader, val_loader, test_loader, normalisation_embedding, d090

##############################################################################
# TO CREATE THE DATASET FOR TRAINING
##############################################################################
class SemiSupervisedImagePairDataset(Dataset):
    
    def __init__(self, root_dir, class_names_to_idx, model_DML, 
                 transform=None, transform_DML=None, probabilityUsePrior=0.0, 
                 indicesAPN=None, boolsAPN=None, prob_sampleforceDMLprior_pairs=0.0,
                 prob_sampleforce_DEEM_pairs=0.0,indexingEmbeddings=None):
        
        ##############################
        # !!!!!!!!!!!!!!!!!!!
        # IF A CHANGE IS MADE BELOW CHANGE ALSO THE SPLIT FUNCTION
        ##############################
        
        self.root_dir = root_dir # path to images
        self.transform = transform if transform is not None else transforms.ToTensor() # transform to apply
        self.transform_DML = transform_DML # if transform_DML is not None else transforms.ToTensor()
        self.image_paths = self._get_image_paths() # sorted names
        self.class_names_to_idx = class_names_to_idx # class index
        self.labels = self._get_labels() # labels
        self.model_DML = model_DML # DML model
        self.probabilityUsePrior = probabilityUsePrior # set the number of labeled points
        self.prob_sampleforceDMLprior_pairs = prob_sampleforceDMLprior_pairs # labeled pairs from DML
        #self.indexes_prior1, self.indexes_prior2 = self._get_prior_indices() 
        self.indexes_prior = self._get_prior_indices() # version where a vector of true / false indicates if the data is labeled, x% are labeled according to self.probabilityUsePrior 
        self.true_indices = [index for index, value in enumerate(self.indexes_prior) if value]
        self.indexingEmbeddings = indexingEmbeddings # dictionary that link image name to embedding
        self.prob_sampleforce_DEEM_pairs = prob_sampleforce_DEEM_pairs           
        # sanity check about prior
        # assert(int(args['useDMLprior_pairs2labels'] * args['useDMLprior_pairs2labels'] == 0)==0)
        # assert(args['useDMLprior_pairs'] == 1 or args['useDMLprior_pairs'] == 0)
        # assert(args['useDMLprior_pairs2labels'] == 1 or args['useDMLprior_pairs2labels'] == 0)
        # assert(self.probabilityUsePrior >= 0.0 and self.probabilityUsePrior <= 1.0)
        # assert(self.prob_sampleforceDMLprior_pairs >= 0.0 and self.prob_sampleforceDMLprior_pairs <= 1.0)       
        # if self.prob_sampleforceDMLprior_pairs > 0.0: # vice versa
        #     assert(args['useDMLprior_pairs'] == 1 or args['useDMLprior_pairs2labels'] == 1)            
        #if args['useDMLprior_pairs'] == 1 or args['useDMLprior_pairs2labels'] == 1:
        #    assert(self.prob_sampleforceDMLprior_pairs > 0.0)            
         
        # sanity check about pairs 
        # Preprocess indicesAPN into a list and a dict for efficient access
        # self.list_supervised_pairs is a list of sorted tuples, allowing efficient random sampling without repeated conversions.
        # Dictionary for Lookup: self.dict_supervised_pairs maps each sorted tuple pair to its corresponding index in boolsAPN, facilitating quick
        # similarity checks. By maintaining self.list_supervised_pairs, we avoid the overhead of converting a set to a tuple on every call 
        # to __getitem__. Instead, we can directly use random.choice on the precomputed list.
        self.check_raw_data(indicesAPN, "Before frozenset processing")  # Check the raw data before processing        
        self.list_supervised_pairs, self.dict_supervised_pairs, self.reverse_index = self.preprocess_indicesAPN(indicesAPN)
        self.check_processed_data(self.dict_supervised_pairs, "After frozenset processing")  # Check the processed data after conversion
        self.boolsAPN = boolsAPN  # true if similar, false if dissimilar
        
    # load a file where key are file name followed by embeddings
    def _load_indexingEmbeddings(self):
        return []
    
    # Preprocess indicesAPN into a set of pairs
    def preprocess_indicesAPN(self, indicesAPN):
        """
        Converts indicesAPN into a list of sorted tuples for random sampling
        and a dictionary for quick lookup.
        """
        if indicesAPN is None:
            return [], {}, []
        else:
            # Convert each pair to a sorted tuple
            list_supervised_pairs = [tuple(sorted(pair.tolist())) for pair in indicesAPN]
            
            # Create a dictionary mapping each pair to its index
            dict_supervised_pairs = {pair: idx for idx, pair in enumerate(list_supervised_pairs)}
            
            # Create a reverse index mapping each index to its associated pairs
            reverse_index = defaultdict(list)
            for pair in list_supervised_pairs:
                reverse_index[pair[0]].append(pair)
                reverse_index[pair[1]].append(pair)

        return list_supervised_pairs, dict_supervised_pairs, reverse_index
        
    # sort files and dirs to ensure consistency concerning labeled indices    
    def _get_image_paths(self):
        image_paths = []
        for subdir, _, files in sorted(os.walk(self.root_dir)):
            for file in sorted(files):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(subdir, file))
        return image_paths

    def _get_labels(self):
        labels = []
        for path in self.image_paths:
            label = os.path.basename(os.path.dirname(path))
            label = self.class_names_to_idx[label]
            labels.append(label)
        return labels

    def _get_prior_indices(self):
        # Bernoulli sampling to decide prior indices
        prior_indices = torch.bernoulli(torch.full((len(self.labels),), self.probabilityUsePrior)).bool()
        return prior_indices

    def check_raw_data(self, indicesAPN, message):
        # Check if all elements in indicesAPN are pairs before processing
        if indicesAPN is not None:
            print(f"\n--- {message} ---")
            invalid_entries = []
            for i, pair in enumerate(indicesAPN):
                if len(pair) != 2:
                    invalid_entries.append((i, pair))
                if i < 5:  # Print the first 5 valid entries for inspection
                    print(f"Raw pair {i}: {pair}")
            if invalid_entries:
                print(f"Found {len(invalid_entries)} invalid entries in raw indicesAPN (should be pairs):")
                for idx, entry in invalid_entries:
                    print(f"Index {idx}: {entry}")
                raise 'Probably a double entry'
            else:
                print("All entries in raw indicesAPN are valid pairs.")
    
    def check_processed_data(self, indicesAPN, message):
        # Check the processed data after conversion to frozenset
        if indicesAPN is not None:
            print(f"\n--- {message} ---")
            invalid_entries = []
            for i, pair in enumerate(indicesAPN):
                if len(pair) != 2:
                    invalid_entries.append((i, pair))
                if i < 5:  # Print the first 5 valid entries for inspection
                    print(f"Processed pair {i}: {pair}")        
            if invalid_entries:
                print(f"Found {len(invalid_entries)} invalid entries in processed indicesAPN (should be pairs):")
                for idx, entry in invalid_entries:
                    print(f"Index {idx}: {entry}")
                raise 'Probably a double entry'
            else:
                print("All entries in processed indicesAPN are valid pairs.")

    def __len__(self):
        return len(self.image_paths)
            
    # key funcion to sample pairs
    def __getitem__(self, idx):

        try:
            idx1 = idx
            
            # Decide how to sample idx2: if we use pairs and proba of sample a labeled pair used in DML > thresh
            if random.random() < self.prob_sampleforceDMLprior_pairs and self.list_supervised_pairs:
                # Use reverse_index for fast candidate selection
                candidates = self.reverse_index[idx]
                if candidates: #print('I found idx in reverse list')
                    selected_pair = random.choice(candidates)
                    idx1, idx2 = selected_pair # this is a pair that was used in DML as a labeled pair
                    # Ensure correct ordering of idx1
                    if idx1 != idx:
                        idx1, idx2 = idx2, idx1

                else: #print('I used random because can not fin idx in reverse list')
                    # Fall back to random sampling
                    idx2 = random.randint(0, len(self.image_paths) - 1)
                    while idx2 == idx1:
                        idx2 = random.randint(0, len(self.image_paths) - 1)
            
            elif self.probabilityUsePrior > 0.0: # case with prior in DEEM              
                # If img1 is labeled the draw img2 as labeled with probably p 
                # so that the pair is labeled otherwise draw a random pair            
                if self.indexes_prior[idx1]:
                    if random.random() < self.prob_sampleforce_DEEM_pairs:
                        # Draw img2 in the list of labeled images
                        idx2 = random.choice(self.true_indices)
                        while idx2 == idx1:
                            idx2 = random.choice(self.true_indices)
                    else:
                        # Draw img2 in the whole dataset
                        idx2 = random.randint(0, len(self.image_paths) - 1)
                        while idx2 == idx1:
                            idx2 = random.randint(0, len(self.image_paths))
                else:
                    # Random sampling logic            
                    idx2 = random.randint(0, len(self.image_paths) - 1)
                    while idx2 == idx1:
                        idx2 = random.randint(0, len(self.image_paths) - 1)
                                    
            else:
                # Random sampling logic            
                idx2 = random.randint(0, len(self.image_paths) - 1)
                while idx2 == idx1:
                    idx2 = random.randint(0, len(self.image_paths) - 1)
            
            idx_prior1 = self.indexes_prior[idx1]
            img1_path = self.image_paths[idx1]
            label1 = int(self.labels[idx1])
            
            idx_prior2 = self.indexes_prior[idx2]
            img2_path = self.image_paths[idx2]
            label2 = int(self.labels[idx2])
                        
            # Check if paths are valid
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                print(f"🚨 Missing image: {img1_path} or {img2_path}")
                return None
                
            # We extract the embeddings from files name here, but for DML this is done in the train loop, not ideal but it is as it is...
            if self.indexingEmbeddings is not None: # then extracts the embeddings from file names            
                precompembedding_img1 = torch.tensor(self.indexingEmbeddings[os.path.basename(img1_path)])
                precompembedding_img2 = torch.tensor(self.indexingEmbeddings[os.path.basename(img2_path)])            
            else:
                # Instead of None, return a zero tensor
                precompembedding_img1 = torch.zeros(0)  # Adjust size as needed
                precompembedding_img2 = torch.zeros(0)  # Adjust size as needed
                        
            # Open images
            try:
                img1 = Image.open(img1_path)#.convert('RGB')
                img2 = Image.open(img2_path)#.convert('RGB')
            except Exception as e:
                print(f"🚨 Error loading images: {img1_path} or {img2_path} - {e}")
                return None

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
            # Check if (idx1, idx2) is a supervised pair - if needed
            # this is a boolean that tells the pair is similar or dissimilar and we do not know
            similarDML = -1
            pair = tuple(sorted([idx1, idx2]))
            if pair in self.dict_supervised_pairs:
                similarDML = int(self.boolsAPN[self.dict_supervised_pairs[pair]])
            
            # Determine similarity based on prior labels - if needed
            similarDEEM = -1
            if idx_prior1 and idx_prior2: # both are labeled
                similarDEEM = int(label1 == label2)
                
            # 🚨 Debugging print statements
            #print(f"✔ Returning: {img1.shape}, {img2.shape}, {label1}, {label2}, {idx_prior1}, {idx_prior2}, {similarDEEM}, {similarDML}, {precompembedding_img1}, {precompembedding_img2}")
    
            return img1, img2, label1, label2, idx_prior1, idx_prior2, similarDEEM, similarDML, precompembedding_img1, precompembedding_img2
        
        except Exception as e:
            print(f"🚨 Error in `__getitem__` for index {idx}: {e}")
            return None