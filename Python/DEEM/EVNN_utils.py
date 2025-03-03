import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, contingency_matrix, pair_confusion_matrix
from EVNN_belief import bbas2plcontour
from EVNN_config import NUMCLASSES, device
from collections import deque
from scipy.spatial.distance import cdist

##############################################################################
# TO TAKE INTO ACCOUNT A PRIOR IN THE LOSS
##############################################################################
# take labels into accounts

def accountPrior(criterion, bbas1, bbas2, labels1, labels2, indexes_prior1, indexes_prior2):
    # predicted labels must correspond to the true one
    # equation 21
    
    if indexes_prior1.any():
        # get BBAs        
        preds1 = bbas2plcontour(bbas1)                
        # select labels according to prior, transformed into indices through long()
        labels1 = labels1[indexes_prior1]#.long() 
        labels1 = labels1.to(preds1.device)
        preds1 = preds1[indexes_prior1.squeeze()]
        # convert in one hot for MSE
        labels1 = torch.nn.functional.one_hot(labels1, NUMCLASSES).to(torch.float32)                
        # new loss including labels
        lossPr1 = criterion(preds1,labels1)
    else:
        lossPr1 = 0.0
           
    if indexes_prior2.any():
        # get BBAs
        preds2 = bbas2plcontour(bbas2)                
        # select labels according to prior, transformed into indices through long()
        labels2 = labels2[indexes_prior2]#.long() 
        labels2 = labels2.to(preds2.device)
        preds2 = preds2[indexes_prior2.squeeze()]
        # convert in one hot for MSE
        labels2 = torch.nn.functional.one_hot(labels2, NUMCLASSES).to(torch.float32)                
        # new loss including labels
        lossPr2 = criterion(preds2,labels2)
    else:
        lossPr2 = 0.0
    
    return lossPr1, lossPr2


# """ # Prend en compte les contraintes sur le fait que deux entrées soient de classe similaires
# def accountPrior_constraints_TD_v2(conflictij, priorsimilar):
#     # EVNNpriorsimilar contains 1 for similar labels, 0 for dissimilar labels, -1 else    
#     # Follows EVCLUSNN 2019 but we modify the loss 
#     # For must link (empty=0)
#     # plij(Sij)  = 1-mij(empty) ok same - pl(fact that images are in the same classe) 
#     # Cas 2 classes : masse(meme classe) = m1(w1)m2(w1) + m1(w2)m2(w2) + m1(w1)m2(Omega) + m1(Omega)m2(w1) + m1(w2)m2(Omega) + m1(Omega)m2(w2) + m1(Omega)m2(Omega) = 1 - (m1(0) + m2(0) - m1(0)m2(0) + m1(w1)m2(w2) + m1(w2)m2(w1)) = 1 - Kij
#     # Pour plij(Sijbarre) Thierry propose
#     # plij(Sijbarre) = 1-mi(empty)-mj(empty)+mi(empty)mj(empty)-sum_singelons(mi(single)mj(single))=1-(mi(empty)+mj(empty)-mi(empty)mj(empty)+sum_singelons(mi(single)mj(single)) => ce qui ressemble au cas précédent bizarre
#     # On aurait tendance a écrire que masse(images PAS dans la meme classe) = mi(empty)(somme de tous les elements=1) + mj(empty)(somme=1) - mi(empty)mj(empty) + m1(w1)m2(w2) + m1(w2)m2(w1) donc en gros 1-cas precedent = Kij ?
#     # Ensuite il propose lossij = plij(Sijbarre) + 1-plij(Sij) for each constraint on ij    
#     # Moi j'aurai tendance a prendre 
#     # loss_ML = fonction des plij(Sij) qui doit etre aussi grande que possible donc min de 1-plij(Sij) ou au carré
#     # Et loss_CNL = fonction des plij(Sijbarre) = doit etre la plus grande possible également donc 1-plij(Sijbarre) ou au carré
        
#     # Define the range for singleton indices
#     singletons = torch.arange(1, NUMCLASSES + 1, device=conflictij.device)  # [1, 2, ..., NUMCLASSES]
    
#     loss_const_ml = 0.0
#     loss_const_cnl = 0.0
                
#     # Filter out entries where priorsimilar is not -1
#     mask = priorsimilar != -1

#     if mask.any():
#         similar = priorsimilar == 1  # Must-link
#         if similar.any():
#             plijSij = 1.0 - conflictij[similar]  # 16a
#             loss_const_ml = torch.mean((1.0-plijSij)*(1.0-plijSij))
        
#         dissimilar = priorsimilar == 0  # Cannot-link
#         if dissimilar.any():
#             plijSijbarre = conflictij[dissimilar]  # 16a
#             loss_const_cnl = torch.mean((1.0-plijSijbarre)*(1.0-plijSijbarre))
#     else:
#         loss_const_ml = 0.0
#         loss_const_cnl = 0.0

#     # Return both must-link and cannot-link losses    
#     return loss_const_ml, loss_const_cnl """


# Prend en compte les contraintes sur le fait que deux entrées soient de classe similaires
def accountPrior_constraints_TD(bbas1, bbas2, conflictij, priorsimilar):
    
    #raise "Error"
    # EVNNpriorsimilar contains 1 for similar labels, 0 for dissimilar labels, -1 else    
    # Follows EVCLUSNN 2019
    # For must link
    # plij(Sij)  = 1-mij(empty)
    # plij(Sijbarre) = 1-mi(empty)-mj(empty)+mi(empty)mj(empty)-sum_singelons(mi(single)mj(single))
    # then lossij = plij(Sijbarre) + 1-plij(Sij) for each constraint on ij    
    # See also for Cannot Link
        
    # Define the range for singleton indices
    #positionClasse1 = 1
    #positionClasseK = positionClasse1 + NUMCLASSES - 1
    #singletons = list(range(positionClasse1, positionClasseK + 1))  # Adjusted to include positionClasseK
    singletons = torch.arange(1, NUMCLASSES + 1, device=bbas1.device)  # [1, 2, ..., NUMCLASSES]
    
    loss_const_ml = torch.tensor(0.0)
    loss_const_cnl = torch.tensor(0.0)
    plijSij_ML = torch.tensor(0.0) # doit etre grande 
    plijSij_CNL = torch.tensor(0.0) # doit etre petite
    plijSijbarre_ML = torch.tensor(0.0) # doit etre petite
    plijSijbarre_CNL = torch.tensor(0.0) # doit etre grande 
                
    # Filter out entries where priorsimilar is not -1
    mask = priorsimilar != -1

    if mask.any():
        similar = priorsimilar == 1  # Must-link
        if similar.any():
            plijSij = 1.0 - conflictij[similar]  # 16a
            lastterm = torch.sum(bbas1[similar][:, singletons] * bbas2[similar][:, singletons], dim=1)
            plijSijbarre = 1.0 - bbas1[similar, 0] - bbas2[similar, 0] + bbas1[similar, 0] * bbas2[similar, 0] - lastterm
            loss_const_ml = torch.mean(plijSijbarre + 1.0 - plijSij)
            plijSij_ML = plijSij
            plijSijbarre_ML = plijSijbarre

        dissimilar = priorsimilar == 0  # Cannot-link
        if dissimilar.any():
            # Equation of Denoeux and Antoine : the last term is strange
            plijSij = 1.0 - conflictij[dissimilar]  # 16a
            lastterm = torch.sum(bbas1[dissimilar][:, singletons] * bbas2[dissimilar][:, singletons], dim=1)
            plijSijbarre = 1.0 - bbas1[dissimilar, 0] - bbas2[dissimilar, 0] + bbas1[dissimilar, 0] * bbas2[dissimilar, 0] - lastterm
            loss_const_cnl = torch.mean(plijSij + 1.0 - plijSijbarre)
            plijSij_CNL = plijSij
            plijSijbarre_CNL = plijSijbarre

    else:
        loss_const_ml = 0.0
        loss_const_cnl = 0.0

    # Return both must-link and cannot-link losses    
    return loss_const_ml, loss_const_cnl, torch.mean(plijSij_ML), torch.mean(plijSij_CNL), torch.mean(plijSijbarre_ML), torch.mean(plijSijbarre_CNL)


def cluster_accuracy(y_true, y_pred):
    """
    Compute cluster accuracy using Hungarian algorithm.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    assert len(y_true) == len(y_pred)

    # Create the confusion matrix
    num_classes = int(max(max(y_true), max(y_pred)) + 1)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(y_true)):
        confusion_matrix[y_pred[i].astype(int), y_true[i].astype(int)] += 1

    # Use Hungarian algorithm to find optimal one-to-one mapping
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    # Compute accuracy based on the optimal mapping
    correct_matches = confusion_matrix[row_ind, col_ind].sum()
    total_matches = len(y_true)
    accuracy = correct_matches / total_matches

    return accuracy

def smooth_measure(measure_history, smooth_window, variation_threshold):
    # measurehistory keeps in memory the values of a measures over a sliding window
    # when the relative variation of the measure to the average is less than a threshold 
    # then returns True
    
    should_break = False
    
    # If enough losses have been collected, compute smoothed loss
    if len(measure_history) >= smooth_window:
        # Compute the smoothed loss (moving average)
        smooth_measure = sum(measure_history[-smooth_window:]) / smooth_window
        
        # Compute the variation (percentage change)
        if len(measure_history) >= smooth_window + 1:
            previous_smooth_measure = sum(measure_history[-(smooth_window+1):-1]) / smooth_window
            # Percentage change in loss
            variation = abs(smooth_measure - previous_smooth_measure) / previous_smooth_measure
            
            # Check if the variation is below the threshold
            if variation < variation_threshold:
                print(f'Variation on {smooth_window}: {variation} (thresh={variation_threshold}) (old={previous_smooth_measure}, new={smooth_measure})')
                should_break = True
         
    return should_break

# # Early Stopping Class
# class EarlyStopping:
#     # Example usage
#     # init
#     # early_stopping = EarlyStopping(patience=5, min_delta=0.0025)
#     # then in the training loop
#     # ...
#     # for data in dataloader
#     #       ...
#     #       early_stopping(-val_loss, model)        # call with "-" for loss, because we are checking is an perf indicator is < to previous to evaluate the patience
#     #       or early_stopping(val_accuracy, model)  # if val_acc decrease "patience" times then stop
#     #       if early_stopping.early_stop:    # check
#     #           print("Early stopping")
#     #           break
    
#     def __init__(self, patience=5, min_delta=0.005):
#         assert(min_delta > 0)
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         #self.best_model_wts = None
#         self.N = len(str(min_delta).split('.')[-1])

#     def __call__(self, criterion, model):
#         #score = -val_loss
#         score = criterion
#         if self.best_score is None:
#             self.best_score = score
#             #self.best_model_wts = model.state_dict()
#         elif round( np.abs(score - self.best_score), self.N) <= self.min_delta:
#             self.counter += 1
#             print(f'Early stopping -> crit={round( np.abs(score - self.best_score), self.N)} seuil {self.min_delta}, cpt={self.counter} (pat.={self.patience})')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             #self.best_model_wts = model.state_dict()
#             self.counter = 0

    
class SlidingWindow:
    # Example usage
    # window_size = 1000  # Adjust based on batch size and memory constraints
    # percentile_estimator = SlidingWindowPercentile(window_size)
    # returns  value for percentile required, min, max, 95th and 5th percentiles 
    
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = deque()  # Using deque for efficient pop from left

    def update(self, new_batch):
        # Add new elements from the batch
        for element in new_batch:
            if len(self.data) >= self.window_size:
                self.data.popleft()  # Remove the oldest element            
            if isinstance(element, torch.Tensor): # Ensure element is on CPU and converted to a Python scalar if it's a tensor
                element = element.cpu().item() 
            self.data.append(element)
    
    def getdata(self):
        return list(self.data)
    
    #def calculate_percentile(self, percentile):
    #    if not self.data:
    #        return None
    #    return np.percentile(list(self.data), percentile), np.min(list(self.data)), np.percentile(list(self.data), 10)


class EarlyStopping:
    def __init__(self, window_size, patience, tolerance_ratio=0.1, min_val=0.05):
        """
        Args:
        - window_size: int, number of iterations for the sliding window.
        - patience: int, number of consecutive iterations required to trigger early stopping.
        - tolerance_ratio: float, tolerance around the mean (as a percentage).
        - min_gradient_norm: float, minimum value required before stopping
        Principle: µ-(µxr) <= grad_norm <= µ+(µxr) => If grad_norm is within this range, it is considered "close" to the mean.
        """
        self.window_size = window_size
        self.patience = patience
        self.tolerance_ratio = tolerance_ratio
        self.gradient_norms = deque(maxlen=window_size)
        self.counter = 0
        self.min_val = min_val

    def update(self, grad_norm):
        """
        Updates the norms and checks the stopping criterion.
        Args:
        - grad_norm: float or tensor, current gradient norm, loss, or any quantity to minimize.
        Returns:
        - bool, True if stopping is triggered, otherwise False.
        """
        # Ensure grad_norm is a CPU float before adding to deque
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.cpu().item()
        
        self.gradient_norms.append(grad_norm)

        # If the sliding window is not yet full, do not check the stopping criterion
        if len(self.gradient_norms) < self.window_size:
            return False

        # Compute the mean and the tolerance interval
        mean_grad = np.mean(list(self.gradient_norms))
        tolerance = mean_grad * self.tolerance_ratio

        # Check if the norm is within the tolerance interval
        if abs(grad_norm - mean_grad) <= tolerance and mean_grad < self.min_val:
            self.counter += 1
        else:
            self.counter = 0  # Reset the counter if out of the interval or too large

        # Check if the patience threshold is reached
        if self.counter >= self.patience:
            print(f"Early stopping triggered: Mean = {mean_grad:.6f}, Current Norm = {grad_norm:.6f}")
            return True

        return False





def is_pair_present(idx, idx2, indicesAPN_set):
    # Use frozenset for efficient lookup in the set
    return frozenset((idx, idx2)) in indicesAPN_set

def get_pair_index(idx, idx2, indicesAPN):
    target_set = frozenset([idx, idx2])
    if target_set in indicesAPN:
        return list(indicesAPN).index(target_set)  # Return the index of the found pair
    return None

""" def get_pair_index(idx, idx2, supervised_indices_tensor):
    
    if supervised_indices_tensor is not None:
        # Check if (idx, idx2) or (idx2, idx) is present in the Nx2 tensor
        pair_1 = (idx, idx2)
        pair_2 = (idx2, idx)
        
        # Convert tensor to a list of tuples for easier comparison
        pairs_list = [tuple(row) for row in supervised_indices_tensor]
        
        # Check if pair_1 or pair_2 is present and return its index
        if pair_1 in pairs_list:
            return pairs_list.index(pair_1)
        elif pair_2 in pairs_list:
            return pairs_list.index(pair_2)
    
        return None
    
    else:
        # If neither pair is found, return None or another appropriate value
        return None """
        
        

def compute_90th_percentile_matrix(embeddings, sample_size=1000):
    """
    Compute the 95th percentile of pairwise distances using matrix operations.
    Suppose the same distance as in DEEM !!!
    Suppose the same percentile as in DEEM !!!
    
    Args:
        embeddings (torch.Tensor): The dataset of embeddings (n_samples, n_features).
        sample_size (int): Number of embeddings to sample for estimating the percentile.
        device (str): Device to use ("cpu" or "cuda").

    Returns:
        float: The 90th percentile of pairwise distances.
    """
    # Randomly sample embeddings
    #embeddings = embeddings.to(device)
    sampled_indices = torch.randperm(len(embeddings))[:sample_size]
    sampled_embeddings = embeddings[sampled_indices]

    # Compute pairwise distance matrix
    distances_matrix = torch.cdist(sampled_embeddings, sampled_embeddings, p=2)  # (sample_size, sample_size)

    # Extract upper triangle (unique distances)
    distances = distances_matrix[torch.triu(torch.ones_like(distances_matrix), diagonal=1) == 1]

    # Compute the 95th percentile
    percentile_90 = torch.quantile(distances, 0.90).item()

    return percentile_90



############################################################################
# Output matrix "C" used to compute conflict between two masses
############################################################################

def get_Cmatrix_and_forDecision(numClasses):
        
        # We parameterise the output layer
        npairesmasses = numClasses*(numClasses-1)/2
        usedoute = 1
        usepaires = 1
        sizeLastLayer = numClasses + 1 + usedoute + usepaires*npairesmasses    
        options_DEEM = {
                                "numClasses": int(numClasses),
                                "usedoute": int(usedoute),
                                "usepaires": int(usepaires),
                                "fs": int(sizeLastLayer),
                                "npairesmasses": int(npairesmasses),
                                "pairCoding": [] 
                            }
        options_DEEM['pairCoding'] = build_pairCoding(options_DEEM)    
        Cmatrix, for_decision = buildCmatrix22(options_DEEM)                  
        for_decision = torch.tensor(for_decision, dtype=torch.float32).to(device=device)
        Cmatrix = torch.tensor(Cmatrix, dtype=torch.float32).to(device=device)
        return Cmatrix, for_decision
    
    
##############################################################################
# Returns two matrices used to compute conflict and plausibility 
##############################################################################

def buildCmatrix22(options_DEEM):
    """    
    options_DEEM should be a dict with keys (at least):
      - "numClasses": int
      - "usedoute": bool
      - "usepaires": bool
      - If usepaires==True:
          * "fs": int  (dimension of the final C matrix)
          * "pairCoding": 2D array of shape (number_of_pairs, 2) 
            (the pairs used in 'ismember' checks)
    Returns:
      - C: 2D numpy array (conflict matrix)
      - forDecision: 2D numpy array or empty array, depending on usepaires
    """

    # Dimension of C is options_DEEM.fs x options_DEEM.fs       
    numClasses = options_DEEM["numClasses"]
    usedoute   = options_DEEM["usedoute"]
    usepaires  = options_DEEM["usepaires"]
    assert(usedoute == 1)
    fs         = options_DEEM["fs"]
    pairCoding = options_DEEM["pairCoding"]  # shape (nbpairs, 2) 
    
    # We'll define these as they are in MATLAB, then subtract 1 later
    positionConflict = 1
    positionClasse1 = 2
    positionClasseK = numClasses + 1
    positionDoute    = numClasses + 2            
        
    # ------------------------------------------------
    # Main logic branch: usepaires / ~usepaires
    # ------------------------------------------------
    if usepaires:
        
        # number of pairs
        nbpaires = (numClasses * (numClasses - 1)) // 2

        # ------------------------------------------------
        # Build the conflict matrix C
        # ------------------------------------------------
        C = np.ones((fs, fs), dtype=int)
        np.fill_diagonal(C, 0)  # Zero out the diagonal % no conflict on diagonal A1 inter A2
        
        C[positionDoute - 1, :] = 0 # for universe intersect diff from 0 no conflict
        C[:, positionDoute - 1] = 0

        C[positionConflict - 1, :] = 1 # for conflict 
        C[:, positionConflict - 1] = 1

        # ------------------------------------------------
        # Intersection paires - singletons
        # ------------------------------------------------
        row_slice = slice(positionDoute, None)  # 0-based

        for i in range(1, numClasses + 1):
            # The column in MATLAB is i+1. In Python's 0-based indexing, it's (i+1) - 1 => i
            col_index = i

            # 1) Build the boolean mask for "rows that contain i"
            #    pairCoding == i => shape (npaires, 2), with True/False
            mask_2d = (pairCoding == i)
            #    Summation along axis=1 => number of Trues in each row
            #    > 0 => row has i
            mask_any = np.sum(mask_2d, axis=1) > 0  # shape (npaires,)

            # 2) We want the *negation* => same as "not(...)"
            #    This yields a boolean vector with True where i is *not* in that row
            vals_bool = ~mask_any
            # If we want floats (0.0/1.0) to store in C:
            vals_float = vals_bool.astype(float)

            # 3) Assign into C:
            #    => C[row_slice, col_index] 
            C[row_slice, col_index] = vals_float

            # 4) Then symmetrical assignment:
            #    C(i+1, positionDoute+1:end) = C(positionDoute+1:end, i+1)'
            #    => row index = i, col slice = row_slice
            #    => but we need the *same vector*, which is vals_float
            C[col_index, row_slice] = vals_float
            
        # ------------------------------------------------
        # Intersection paires - paires
        # ------------------------------------------------
                
        row_slice = slice(positionDoute, None)  # 0-based
        for i in range(nbpaires):
            col_index = positionDoute + i
            pair_i = pairCoding[i, :]                  # shape (2,)
            mask_2d = np.isin(pairCoding, pair_i)      # shape (#pairs, 2), True where matching any element
            mask_any = np.sum(mask_2d, axis=1) > 0     # shape (#pairs,), True if row has at least one match
            vals_bool = ~mask_any                      # invert
            vals_float = vals_bool.astype(float)       # we assume C is float
            C[row_slice, col_index] = vals_float
            C[col_index, row_slice] = vals_float
        
        # ------------------------------------------------
        # Build forDecision
        # ------------------------------------------------
        
        # Rows: from positionDoute to the end
        row_slice = slice(positionDoute, None)
        col_slice = slice(positionClasse1 - 1, positionClasseK)
        subC = C[row_slice, col_slice]
        forDecision = (subC == 0).astype(int)
        
    else:
        # ------------------------------------------------
        # else branch: *no* paires
        # ------------------------------------------------
        # C is (numclasses + 2*(usedoute)) x (numclasses + 2*(usedoute))
        extra = 2 if usedoute else 0
        length = numClasses + extra

        C = np.ones((length, length), dtype=float)
        np.fill_diagonal(C, 0.0)

        if usedoute:
            # same logic as above
            C[positionDoute - 1, :] = 0
            C[:, positionDoute - 1] = 0
            C[positionConflict - 1, :] = 1
            C[:, positionConflict - 1] = 1

        # forDecision is empty in this branch
        forDecision = np.array([], dtype=int)

    return C, forDecision


##############################################################################
# Returns pairs, used in C-matrix
##############################################################################

def build_pairCoding(options_DEEM):
    """
    options_DEEM is a dict with:
       - 'npairesmasses' (int): the total number of pairs
       - 'numClasses' (int): total number of classes
    We'll create a (npairesmasses x 2) array of pairs [i, j].
    """
    numClasses = options_DEEM["numClasses"]
    npairesmasses = options_DEEM["npairesmasses"]

    # Initialize an empty NumPy array of shape (npairesmasses, 2)
    pairCoding = np.zeros((npairesmasses, 2), dtype=int)

    k = 0  # Note that Python uses 0-based indexing
    for i in range(1, numClasses + 1):
        for j in range(i + 1, numClasses + 1):
            pairCoding[k, :] = [i, j]
            k += 1

    # Store in options_DEEM if you like
    options_DEEM["pairCoding"] = pairCoding

    return pairCoding


    
    