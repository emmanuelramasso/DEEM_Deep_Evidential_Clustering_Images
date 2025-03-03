import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from tqdm.auto import tqdm
import json
import argparse
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

from Contrastive_DML import *

# DEGUG
#import cProfile
#import pstats
#import torch.autograd.profiler as profiler


def load_hyperparameters_from_file(file_path):
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)
        return hyperparameters


def main():
    
    # Load the json file
    parser = argparse.ArgumentParser()
    parser.add_argument('-json', '--jsonfile', required=True, type=str, help='Path to json config filename')
    argsJson = vars(parser.parse_args())
    config_file = argsJson['jsonfile']
    args = load_hyperparameters_from_file(config_file)    
    print(args)
    
    dataset_name    = args['dataset_name']
    imagesPath      = args['path_to_images']
    modelnamefile   = args['modelnamefile'] # 'semisup_0.25_ep10_model_metric_learning_MNIST'
    supervised_prob = args['supervised_prob'] # 0.25
    num_epochs      = args['num_epochs'] #20 # 50
    batch_size      = args['batch_size']
    output_shape    = args['output_shape']
    evaluation_interval = args['evaluation_interval']

    #assert(0<args['training_split_percentage']<=100)
    if os.path.exists(modelnamefile+'.pth'):
        raise FileExistsError(f"The file {modelnamefile} already exists.")
    if os.path.exists(modelnamefile+'_repo'):
        raise FileExistsError(f"The repository {modelnamefile+'_repo'} already exists.")

    dest_dir = modelnamefile+"_repo"
    os.makedirs(dest_dir, exist_ok=True)

    model_path = modelnamefile+'.pth'
    pathlabeledidxfile = dest_dir+'/'+modelnamefile+'_labeled_file_paths.txt'
    print(pathlabeledidxfile)

    writer = SummaryWriter(log_dir='logdir_tb_dml',filename_suffix=args['modelnamefile'])
    
    #################################################
    # Data augmentation
    # TO be changed for ORION dataset / without grayscale
    print(f'Load dataset {dataset_name}')
    if dataset_name == 'mnist': 
        in_channels = 1
        transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
                transforms.Grayscale(), 
                transforms.ToTensor(),
        ])    
        transformAnchor = transforms.Compose([
                    transforms.Grayscale(), 
                    transforms.ToTensor(),
                ])
    else: 
        raise "unknown dataset"

    use_cuda = torch.cuda.is_available()    
    if use_cuda:
        device = torch.device("cuda")
        idcuda = 0
        print(f"Device name: {torch.cuda.get_device_name(idcuda)}\n")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print('NO GPU !')

    #################################################

    print('Preparing datasets...')
    # DETERMINISTIC BEHAVIOR / sorted filenames versus datasets.ImageFolder(imagesPath) 
    # we sort the images by name, we sample indices and store them
    dataset = SortedImageDataset(imagesPath) 

    # Define labeled samples within the training dataset only
    num_labeled = int(supervised_prob * len(dataset))
    labeled_indices = np.random.choice(len(dataset), num_labeled, replace=False)  # Generate random indices
    print(f'Nb elts for prior p={supervised_prob}: {len(labeled_indices)}')
    print(f'train size: {len(dataset)}')
    
    # Now create the combined datasets with supervision defined
    train_combined_dataset = CombinedSemiSupervisedTripletDataset(dataset, transform, transformAnchor, labeled_indices, supervised_prob, pathlabeledidxfile)

    # DataLoader setup
    batch_size = args['batch_size']
    # better to not shuffle for consistency concerning labeled indices, the randomness is ensured by the labels
    train_loader = DataLoader(train_combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # define model
    print('Loading model...')
    model = SimpleCNN(in_channels).to(device=device)
    if dataset_name == 'mnist':   
        print('load a simple CNN')
        model = SimpleCNN(in_channels).to(device=device)
    else:
        raise "unknown dataset"
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print('CYCLIC LR')            
    #step_size_up = 1000
    #step_size_down = 1000
    #base_learning_rate = 1e-2
    #eta_min = 1e-4
    #print('Base LR (eta_max) : ', base_learning_rate)
    #print('Min LR (eta_min) : ', eta_min)            
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #        optimizer, T_0=2000, T_mult=1, eta_min=eta_min, last_epoch=-1)
    total_iters = float('inf') # 1000
    gamma = 1.0    
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=gamma, total_iters=total_iters)

    # Training loop
    num_samples = 100000
    prct90th = []
    # List to store the loss
    loss_history = [] 
    smooth_window = 5 # Parameters for smoothness check, x epochs min
    variation_threshold = 1/20000 # percentage
    should_break = False
    #indicesAPN_all = torch.tensor([]) 
    #bools_APN_all = torch.tensor([])  
    # Initialize RowManager
    row_manager = RowManager()
    iter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        nbbatchs=0
    
        #with profiler.profile(use_cuda=True) as prof:
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                    
            anchors, positives, negatives, _, _, _, indicesAPN, bools_APN = data
            anchors = anchors.float().to(device=device)
            positives = positives.float().to(device=device)
            negatives = negatives.float().to(device=device)
            indicesAPN = indicesAPN.view(-1, 2) 
            num_elements = bools_APN.numel()  # Total number of elements in the tensor
            k = num_elements // 3  # Number of complete packets of 3
            bools_APN = bools_APN.view(k*3, 1)
                
            # store indices, witouout redondance
            # indices are taken AFTER sorting the file names of data that's key
            # indicesAPN_all contains pairs AP, AN, PN 
            # bools_APN_all contains bool True, False, False
                
            # Convert tensors to NumPy arrays for processing 
            B_numeric = indicesAPN.cpu().numpy()
            B_bool = bools_APN.cpu().numpy()
            # Update rows using RowManager
            row_manager.add_new_rows(B_numeric, B_bool)
                                    
            optimizer.zero_grad()
                
            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)
                
            loss = triplet_loss(anchor_emb, positive_emb, negative_emb, margin=args['margin'])
                
            reg =  torch.mean( torch.sum ( anchor_emb**2, dim=1 ))
            reg += torch.mean( torch.sum ( negative_emb**2, dim=1 ))
            reg += torch.mean( torch.sum ( positive_emb**2, dim=1 ))
            reg = reg/3*0.25*0.05
            loss = loss + reg
                
            writer.add_scalar("Loss", loss, iter)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], iter)
                
            loss.backward()
            optimizer.step()
            scheduler.step()
                
            epoch_loss += loss.item()
            iter += 1
            nbbatchs += 1
        
        epoch_loss = epoch_loss / nbbatchs
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, LR: {scheduler.get_last_lr()}')
        loss_history.append(epoch_loss)
        
        model_path2 = f'DML_model_epoch_{epoch:04d}.pth'
        model_path2 = dest_dir+'/'+model_path2
        scripted_model = torch.jit.script(model)
        scripted_model.save(model_path2)
        print(f"Model DML saved at iteration {epoch} to {model_path2}")

        torch.save({'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch}, dest_dir+'/'+ f'checkpoint_{epoch:04d}.pth')
        # to resume 
        # checkpoint = torch.load('checkpoint.pth')
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch']

        if not should_break and smooth_loss(loss_history, smooth_window, variation_threshold) == True:
            print(f"BREAK AT EPOCH {epoch} (variation small)")
            should_break = True
            break
        
        # Evaluate the model
        if (epoch + 1) % evaluation_interval == 0:
            #print('Evaluate on training data...') 
            #evaluate_model_meth0dist(model, train_combined_dataset, train_loader, device,  num_samples=100000)
            
            #print('Evaluate on testing data...') 
            #evaluate_model_meth0dist(model, test_combined_dataset, test_loader, device,  num_samples=100000)
            
            print('Evaluate many couples on training...') 
            same_label_dists, diff_label_dists, labels_pos, labels_neg, running_mean, running_stddev = \
                evaluate_model_meth0dist_all_pairs(model, train_loader, device=device, num_pairs=20000)
            plot_histograms(same_label_dists, diff_label_dists, dest_dir)
            
            all_dists = same_label_dists + diff_label_dists  
            print(f'size of same {len(same_label_dists)}, size of diff {len(diff_label_dists)}, size tot {len(all_dists)}')
            nth_percentile = 90
            percentile_value = np.percentile(all_dists, nth_percentile)
            print(f"The {nth_percentile}th percentile of the combined distributions is: {percentile_value}, and the stored values are")
            prct90th.append(percentile_value)
            print(prct90th)        
        
        
    #####################
    # After training, retrieve the final lists if needed
    indicesAPN_all, bools_APN_all = row_manager.get_final_lists()
    indicesAPN_all = torch.tensor(indicesAPN_all)
    bools_APN_all = torch.tensor(bools_APN_all)
    
    #####################
    print('Final: Evaluate many couples on training...') 
    same_label_dists, diff_label_dists, labels_pos, labels_neg, running_mean, running_stddev = \
        evaluate_model_meth0dist_all_pairs(model, train_loader, device=device, num_pairs=20000)
    plot_histograms(same_label_dists, diff_label_dists, dest_dir+'/'+modelnamefile+'_histofinal.png')
            
    all_dists = same_label_dists + diff_label_dists  
    print(f'size of same {len(same_label_dists)}, size of diff {len(diff_label_dists)}, size tot {len(all_dists)}')
    nth_percentile = 90
    percentile_value = np.percentile(all_dists, nth_percentile)
    print(f"The {nth_percentile}th percentile of the combined distributions is: {percentile_value}, and the stored values are")
    prct90th.append(percentile_value)
    print(prct90th)

    #####################
    print('Generate embeddings...') 
    embeddings = []
    for anchors, _, _, labels_anchors, _, _, _, _ in train_loader:
        anchors = anchors.float().to(device)
        with torch.no_grad():
            anchor_emb = model(anchors)
        embeddings.append(anchor_emb.cpu().numpy())

    # Flatten embeddings and labels
    embeddings = np.concatenate(embeddings, axis=0)
    mean_embeddings = np.mean(embeddings, axis=0)
    std_embeddings = np.std(embeddings, axis=0)
        
    #####################
    print("Saving...")

    # model
    scripted_model = torch.jit.script(model)
    scripted_model.save(dest_dir+'/'+model_path)

    # input shape in a json
    image, label = dataset[0] 
    input_shape = tuple(transform(image).shape)
    print(input_shape)
    import json
    input_shape_dict = {'input_shape': input_shape}
    with open(dest_dir+'/'+modelnamefile+'_input_shape.json', 'w') as f:
        json.dump(input_shape_dict, f)

    # percentiles in a json
    percentiles = {'prct90th_alldist':  np.percentile(all_dists, 90),        'prct10th_alldist':  np.percentile(all_dists, 10),        'min_alldist':  np.min(all_dists),               'max_alldist': np.max(all_dists), 
                'prct1th_alldist':  np.percentile(all_dists, 1),          'prct99th_alldist':  np.percentile(all_dists, 99),        'prct5th_alldist':  np.percentile(all_dists, 5),     'prct95th_alldist':  np.percentile(all_dists, 95),
                'prct90th_samedist': np.percentile(same_label_dists, 90), 'prct10th_samedist': np.percentile(same_label_dists, 10), 'min_samedist': np.min(same_label_dists), 'max_samedist': np.max(same_label_dists), 
                'prct90th_diffdist': np.percentile(diff_label_dists, 90), 'prct10th_diffdist': np.percentile(diff_label_dists, 10), 'min_diffdist': np.min(diff_label_dists), 'max_diffdist': np.max(diff_label_dists)}
    with open(dest_dir+'/'+modelnamefile+'_percentiles_dict.json', 'w') as f:
        json.dump(percentiles, f)

    # save some vectors
    torch.save(running_mean, dest_dir+'/'+modelnamefile+'_running_mean.pt')
    torch.save(running_stddev, dest_dir+'/'+modelnamefile+'_running_stddev.pt')
    torch.save(labeled_indices, dest_dir+'/'+modelnamefile+'_labeled_indices.pt')
    torch.save(mean_embeddings, dest_dir+'/'+modelnamefile+'_batch_mean.pt')
    torch.save(std_embeddings, dest_dir+'/'+modelnamefile+'_batch_stddev.pt')

    torch.save(indicesAPN_all, dest_dir+'/'+modelnamefile+'_indicesOfPairs.pt')
    torch.save(bools_APN_all, dest_dir+'/'+modelnamefile+'_boolSimilarOfPairs.pt')

    # Define the destination directory and move files to the directory
    # by convention "name of model" + "a certain name" that is used in DEEM 
    # shutil.move(model_path, dest_dir)
    # shutil.move(modelnamefile+'_input_shape.json', dest_dir)
    # shutil.move(modelnamefile+'_percentiles_dict.json', dest_dir)
    # shutil.move(modelnamefile+'_running_mean.pt', dest_dir)
    # shutil.move(modelnamefile+'_running_stddev.pt', dest_dir)
    # shutil.move(modelnamefile+'_batch_mean.pt', dest_dir)
    # shutil.move(modelnamefile+'_batch_stddev.pt', dest_dir)
    # shutil.move(modelnamefile+'_labeled_indices.pt', dest_dir)
    # shutil.move(pathlabeledidxfile, dest_dir)
    # print("Files moved to the repository directory.")

    #####################

    print("Training completed!")
 
 
 
if __name__ == '__main__':
    #profiler = cProfile.Profile()
    #profiler.enable()
    
    main()
    
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('time')
    #stats.print_stats(10)  # Top 10 functions
