# load images and test a model

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import argparse
from Contrastive_DML import *


use_cuda = torch.cuda.is_available()    
if use_cuda:
    device = torch.device("cuda")
    idcuda = 0
    print(f"Device name: {torch.cuda.get_device_name(idcuda)}\n")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print('NO GPU !')

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True, type=str, help='Dataset name')
parser.add_argument('-m', '--model', required=True, type=str, help='Path to model file')
parser.add_argument('-o', '--outputdir', required=True, type=str, help='Output directory')
parser.add_argument('-f', '--outputfile', required=True, type=str, help='Output results file (appended)')
parser.add_argument('-p', '--prior', required=True, type=float, help='Prior prob')
parser.add_argument('-l', '--trainpath', required=True, type=str, help='Path to training set (used in DML training)')
parser.add_argument('-t', '--testpath', required=True, type=str, help='Path to testing set')
argsJson = vars(parser.parse_args())
dataset_name = argsJson['dataset']
modelnamefile = argsJson['model']
output_dir = argsJson['outputdir']
output_file = argsJson['outputfile']
priorqty = argsJson['prior']
trainpath = argsJson['trainpath']
testpath = argsJson['testpath']

#output_mean_file = os.path.join(output_dir, 'tmp_mean_test_DML.pt')
#output_std_file = os.path.join(output_dir, 'tmp_std_test_DML.pt')
output_file = os.path.join(output_dir, output_file)
print(f'Results in {output_file}')
                           
if dataset_name == 'mnist':
    
    NB_CLASSES = 10
    print(f'Nb classes fixed to {NB_CLASSES}')
    batch_size      = 256
    output_shape    = (28,28) 

    # Load the pre-trained model
    print(f'Load model DML from {modelnamefile}')
    model = torch.jit.load(modelnamefile).to(device)
    model.eval()

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

    datasetTrain = datasets.ImageFolder(trainpath) #,transform=transform)
    datasetTest = datasets.ImageFolder(testpath) #,transform=transform)

else:
    raise "error: dataset name unknown"

## Assuming dataset, transform, transformAnchor are already defined
train_combined_dataset = CombinedSemiSupervisedTripletDataset(datasetTrain, transform, transformAnchor, set(), 1, None)  # Assuming full supervision for testest_loader = DataLoader(test_combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_combined_dataset = CombinedSemiSupervisedTripletDataset(datasetTest, transform, transformAnchor, set(), 1, None)  # Assuming full supervision for test
train_loader = DataLoader(train_combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#################################################
# plot histograms, compute distances first
print('Evaluation des distances pour statistiques...')
if True:
    same_label_dists, diff_label_dists, labels_pos, labels_neg, running_mean, running_stddev = \
    evaluate_model_meth0dist_all_pairs(model, test_loader, device=device, num_pairs=1000)

    # histogram in raw format
    plot_histograms(same_label_dists, diff_label_dists, os.path.join(output_dir, 'hist_same_diff_full.png'))

    # histogram after 1-exp norm
    plot_histograms(1.0-np.exp(-1.2*np.array(same_label_dists)), 1.0-np.exp(-1.2*np.array(diff_label_dists)), os.path.join(output_dir, 'hist_1-exp1.2.png'))
    plot_histograms(1.0-np.exp(-0.5*np.array(same_label_dists)), 1.0-np.exp(-0.5*np.array(diff_label_dists)), os.path.join(output_dir, 'hist_1-exp0.5.png'))
    plot_histograms(1.0-np.exp(-3*np.array(same_label_dists)), 1.0-np.exp(-3*np.array(diff_label_dists)), os.path.join(output_dir, 'hist_1-exp3.0.png'))

    # histogram after linear norm
    list1 = np.array(same_label_dists)  # Convert list1 to numpy array (if it's not already)
    list2 = np.array(diff_label_dists)  # Convert list2 to numpy array (if it's not already)
    L = np.concatenate([list1, list2]) # Combine the two lists
    dmin = np.percentile(L, 1)  # 10th percentile
    dmax = np.percentile(L, 99)  # 90th percentile
    print([dmin,dmax,np.percentile(L, 10),np.percentile(L, 90)])
    L_normalized = (L - dmin) / (dmax - dmin) # Normalize the values using the formula with approximate min and max
    #L_normalized[L_normalized >= 1] = 0.99 # Clip values to be within 0.01 and 0.99
    #L_normalized[L_normalized <= 0] = 0.01 # Retrieve L1 and L2 from L_normalized
    # Apply the same sampling for values below 0 and above 1
    L_normalized[L_normalized <= 0] = np.random.uniform(0.01,  0.1,  size=(L_normalized <= 0).sum())
    L_normalized[L_normalized >= 1] = np.random.uniform(0.90,  0.99, size=(L_normalized >= 1).sum())
    same_label_dists_normalized = L_normalized[:len(list1)]  # First len(list1) values for L1
    diff_label_dists_normalized = L_normalized[len(list1):]  # Remaining values for L2
    plot_histograms(same_label_dists_normalized, diff_label_dists_normalized, os.path.join(output_dir, 'hist_linear.png'))

# Generate embeddings and labels
print('Generate embeddings and labels from train dataset...') 
print(f'Size of train {len(train_combined_dataset)}')
embeddings = []
model.eval()
labels = []
for anchors, _, _, labels_anchors, _, _, _,_ in train_loader:
    anchors = anchors.float().to(device)
    with torch.no_grad():
        anchor_emb = model(anchors)
    embeddings.append(anchor_emb.cpu().numpy())
    labels.append(labels_anchors.numpy())

# Flatten embeddings and labels
embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)
print(embeddings.shape)
print(labels.shape)
#mean_embeddings = np.mean(embeddings, axis=0)
#std_embeddings = np.std(embeddings, axis=0)
np.savetxt(os.path.join(output_dir,'embedding_train.csv'), embeddings, delimiter=',')
np.savetxt(os.path.join(output_dir,'labels_train.csv'), labels, delimiter=',')

#print("Mean of embeddings:", mean_embeddings)
#print("Standard deviation of embeddings:", std_embeddings)
# After computation
#torch.save(mean_embeddings, output_mean_file)
#torch.save(std_embeddings, output_std_file)
#print(f'means and stddev saved in {output_mean_file} and {output_std_file}')

#embeddings = embeddings - mean_embeddings
#embeddings = embeddings / std_embeddings

# Apply KMeans clustering
print('TRAIN Kmeans on train dataset...')
kmeans = KMeans(n_clusters=NB_CLASSES, n_init=10)#, random_state=42)
y_pred = kmeans.fit_predict(embeddings)
ari_train = adjusted_rand_score(labels, y_pred)
ami_train = adjusted_mutual_info_score(labels, y_pred)
print("KMEANS WHOLE TRAIN SET AFTER CONVERGENCE")
print(f"Adjusted Rand Index (ARI): {ari_train}")
print(f"Adjusted Mutual Information (AMI): {ami_train}")
acc_train = cluster_accuracy(labels, y_pred)
print(f"Cluster accuracy (ACC): {acc_train}")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, y_pred)
print(cm)

# Now on test set
# Generate embeddings and labels
print('Generate embeddings and labels from test dataset...') 
embeddings = []
model.eval()
labels = []
for anchors, _, _, labels_anchors, _, _, _,_ in test_loader:
    anchors = anchors.float().to(device)
    with torch.no_grad():
        anchor_emb = model(anchors)
    embeddings.append(anchor_emb.cpu().numpy())
    labels.append(labels_anchors.numpy())

# Flatten embeddings and labels
embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)
print(embeddings.shape)
print(labels.shape)
#print('Save test embeddings in CSV...')
#np.savetxt('embedding_test_mnist_0.001m1_trtestval.csv', embeddings, delimiter=',')
#np.savetxt('labels_test_mnist_0.001m1_trtestval.csv', labels, delimiter=',')

# Apply KMeans clustering
print('TEST Kmeans on TEST dataset...')
y_pred = kmeans.predict(embeddings)
ari_test = adjusted_rand_score(labels, y_pred)
ami_test = adjusted_mutual_info_score(labels, y_pred)
print("KMEANS WHOLE TEST SET AFTER CONVERGENCE")
print(f"Adjusted Rand Index (ARI): {ari_test}")
print(f"Adjusted Mutual Information (AMI): {ami_test}")
acc_test = cluster_accuracy(labels, y_pred)
print(f"Cluster accuracy (ACC): {acc_test}")
cm = confusion_matrix(labels, y_pred)
print(cm)

# Append results to output file
with open(output_file, 'a') as f:
    f.write(f"{modelnamefile},{priorqty},{ari_train},{ami_train},{acc_train},{ari_test},{ami_test},{acc_test}\n")


if False:
    
    #################

    print('Evaluate pairs...')
    same_label_dists, diff_label_dists, labels_pos, labels_neg, running_mean, running_stddev = \
        evaluate_model_meth0dist_all_pairs(model, test_loader, device=device, num_pairs=20000)
        
    plot_histograms(same_label_dists, diff_label_dists, 999)
            
    all_dists = same_label_dists + diff_label_dists  
    print(f'size of same {len(same_label_dists)}, size of diff {len(diff_label_dists)}, size tot {len(all_dists)}')
    nth_percentile = 90
    percentile_value = np.percentile(all_dists, nth_percentile)
    print(f"The {nth_percentile}th percentile of the combined distributions is: {percentile_value}")

"""
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('data.csv', delimiter=',', skiprows=1)  # Skip the header if present

# Extract columns (0-indexed)
x = data[:, 0]  # Column 1 for x-axis (optional, could be range)
y1 = data[:, 1]  # Column 2
y2 = data[:, 2]  # Column 3
y3 = data[:, 3]  # Column 4

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='Column 2', marker='o')
plt.plot(x, y2, label='Column 3', marker='s')
plt.plot(x, y3, label='Column 4', marker='^')

# Customize plot
plt.xlabel('X (Column 1)')
plt.ylabel('Values')
plt.title('Columns 2, 3, and 4')
plt.legend()
plt.grid(True)
plt.show()
"""