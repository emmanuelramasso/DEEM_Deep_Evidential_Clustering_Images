# python3 EVNN.py --jsonfile config000001.json 

from __future__ import print_function
import EVNN_config
from EVNN_config import args, device, writer, dest_dir, tbdir

import EVNN_models
from EVNN_models import model_loader
from EVNN_opt import opt_loader 
from EVNN_utils import accountPrior, cluster_accuracy, SlidingWindow, accountPrior_constraints_TD, EarlyStopping
from EVNN_belief import bbas2plcontour

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, contingency_matrix, pair_confusion_matrix
from tqdm.auto import tqdm

from EVNN_datasets import data_loader_DML
from torch.cuda.amp import GradScaler, autocast

# DEGUG
import cProfile
import pstats

##############################################################################
# TO PERFORM TRAINING
##############################################################################

def train(args, model, device, train_loader, val_loader, test_loader, 
          optimizer, scheduler, epoch, nitermax, niterlastphase, d0est, percentile_estimator,
          early_stopping_gradnorm, loss_val_history, best_vals):
    
    # test loader is just here to test in the loop to plot some results, not used for network update
    
    model.train()
    if args['training_mode'] == "DML": 
        train_loader.dataset.model_DML.eval()
        #val_loader.dataset.model_DML.eval()
    
    scaler = GradScaler()
    pdist = nn.PairwiseDistance(p=2)
    
    # WHICH LOSS
    if args['loss_evnn']=='mse':
        criterion1 = nn.MSELoss()
    elif args['loss_evnn']=='bce':
        criterion1 = nn.BCELoss()
    else:
        raise 'unknown loss'
    if args['loss_prior']=='mse':
        criterion2 = nn.MSELoss()
    elif args['loss_prior']=='bce':
        criterion2 = nn.BCELoss()
    elif args['loss_prior']=='huber':       
        criterion2 = nn.HuberLoss()
    else:
        raise 'unknown loss'
    
    stopcrit = False
    
    assert(0.0 < args['log_interval_test'] < 1.0) 
    assert(0.0 < args['log_interval_val'] < 1.0)     
    touslestest = int(args['log_interval_test']*len(train_loader.dataset)/train_loader.batch_size)
    touslesvalidation = int(args['log_interval_val']*len(train_loader.dataset)/train_loader.batch_size) # sert a savoir si on sort ou non
    print(f"Test every {touslestest} iterations")
    print(f"Validation every {touslesvalidation} iterations")
    
    # Run
    npairssupervised = 0
    npairssupervised_similar = 0
    npairssupervised_dissimilar = 0
                
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        images_1, images_2, labels1, labels2, indexes_prior1, indexes_prior2, similarDEEM, similarDML, precompembedding_img1, precompembedding_img2 = data
                        
        iter = (epoch-1)*len(train_loader)+batch_idx+1     
        if iter > nitermax+niterlastphase: #or early_stopping_acc.early_stop == True:
            print(f"NITER MAX REACHED {nitermax} (nitlast = {niterlastphase}) or Early stopping" )
            stopcrit = True
            break              
        
        # Compute embedding with DML or use precomputed embeddings
        if args['training_mode'] == "DML": 
        
            if not hasattr(train_loader.dataset, 'model_DML'): 
                raise AttributeError('Must have a DML model')        
            if train_loader.dataset.transform_DML is None:
                raise ValueError("The DML transform must be provided (even if the size does not change!)")
                
            img1_DML = train_loader.dataset.transform_DML(images_1).to(device)
            img2_DML = train_loader.dataset.transform_DML(images_2).to(device)                                   
        
            with torch.no_grad():
                embedding_i = train_loader.dataset.model_DML(img1_DML)
                embedding_j = train_loader.dataset.model_DML(img2_DML)
                
        elif args['training_mode'] == "PRECOMPUTED_EMBEDDINGS":
            
            embedding_i = precompembedding_img1.to(device)
            embedding_j = precompembedding_img2.to(device)
            
        
        # using training data on DML normalise outputs : not very useful
        #embedding_i = embedding_i - normalisation_embedding['mean']
        #embedding_i = embedding_i / normalisation_embedding['std']
        #embedding_j = embedding_j - normalisation_embedding['mean']
        #embedding_j = embedding_j / normalisation_embedding['std']  
            
        if args['dist_normalisation'] == 'exp':
            ######################
            # NONLINEAR SCALING 1-exp(-gamma*dij)
            ######################
                
            targets = pdist(embedding_i, embedding_j)  
            
            # histogram of distance computed from DML: TD proposed to get the 90th percentiles, for minibatch we update it through iterations
            percentile_estimator.update(targets)            
            d0 = np.percentile(percentile_estimator.getdata(),90)                 
            d0 = 0.95*d0est+0.05*d0 # smooth values
                                        
            # find gamma by optimizer
            #d0 = d0_min + (d0_max - d0_min) * torch.sigmoid(alpha)            
                                     
            gam = -torch.log(torch.tensor(0.05)) / (d0)
            #gam = torch.min(torch.tensor(1.2),gam)
            targets = 1.0 - torch.exp(-gam*(targets))     
                
            if batch_idx % touslesvalidation == 0:
                #print(f'Gamma for iter {iter} = {gam} (d0={d0}) [dmin={d0_min},dmax={d0_max}]')        
                print(f'Gamma for iter {iter} = {gam} (d0={d0})')        
                    
            #reg_alpha = lambda_reg * torch.pow(torch.relu(gamma_min - gamma), 2)     
            #reg_alpha = lambda_reg * torch.pow(alpha, 2)     
                
        #elif args['dist_normalisation'] == 'linear1':
        #    # not very relevant
        #    targets = pdist(embedding_i, embedding_j)                 
        #    dmin = percentiles_dict['prct1th_alldist']
        #    dmax = percentiles_dict['prct99th_alldist']      
        #    targets = (targets - dmin) / (dmax - dmin) # Normalize the values using the formula with approximate min and max               
        #    random_values = torch.empty((targets <= 0.0).sum(),device=device).uniform_(0.01, 0.1)
        #    targets[targets <= 0.0] = random_values # Assign the random values to the appropriate positions in the 'targets' tensor
        #    random_values = torch.empty((targets >= 1.0).sum(),device=device).uniform_(0.90, 0.99)
        #    targets[targets >= 1.0] = random_values 
            
        else:
            raise('??')
        
        # transfer to device and apply EVNN
        images_1, images_2, targets = images_1.to(device, non_blocking=True), images_2.to(device, non_blocking=True), targets.to(device, non_blocking=True)        
        
        optimizer.zero_grad()
        #optimizer_alpha.zero_grad()
        
        # Utilize torch.cuda.amp for Mixed Precision Training
        # Mixed precision training can speed up computations and reduce memory usage by using both 16-bit and 32-bit floating-point types.
        with torch.amp.autocast('cuda'):

            outputs, bbas1, bbas2 = model(images_1, images_2)
            
            ##################################################################
            ############### LOSS related to kij and dij ###############
            ##################################################################
            loss_dijkij = criterion1(outputs, targets)
            
            # Compute gradients 
            #grads = torch.autograd.grad(loss_dijkij, model.parameters(), retain_graph=True)
            #grad_norm_loss_dijkij = sum(g.norm() for g in grads)
            #writer.add_scalar("GradLosses/Gradloss_dijkij", grad_norm_loss_dijkij, iter)
            writer.add_scalar("Losses/Loss_dijkij", loss_dijkij, iter)
            
            ##################################################################
            ###################### PRIOR ######################
            ##################################################################
            # ********* LABELS IN DEEM **********
            # include prior if any, update the loss using hard labels
            lossprior_labels_deem = 0.0
            if args['useDEEMprior_label']==1 and train_loader.dataset.probabilityUsePrior > 0.0: 
                # use labels
                lossPr1, lossPr2 = accountPrior(criterion2, bbas1, bbas2, 
                    labels1, labels2, indexes_prior1, indexes_prior2)                
                
                wsubloss = 0.1
                lossprior_labels_deem = wsubloss*lossPr1 + wsubloss*lossPr2
                writer.add_scalar("Losses/loss_prior_label", lossprior_labels_deem, iter)
            
            # include prior if any, update the loss using pairs           
            lossprior_pairs_deem = 0.0
            if args['useDEEMprior_pair']==1 and train_loader.dataset.probabilityUsePrior > 0.0: 
                # use pairs
                lossConst_ML_deem, lossConst_CNL_deem, plijSij_ML_deem, plijSij_CNL_deem, plijSijbarre_ML_deem, plijSijbarre_CNL_deem = \
                    accountPrior_constraints_TD(bbas1, bbas2, outputs, similarDEEM)
                
                lossprior_pairs_deem = 0.1*lossConst_ML_deem + 0.1*lossConst_CNL_deem
                
                writer.add_scalar("Losses/loss_ML_deem", lossConst_ML_deem, iter)
                writer.add_scalar("Losses/loss_CNL_deem", lossConst_CNL_deem, iter)
                npairssupervised += torch.sum(similarDEEM!=-1)
                npairssupervised_similar += torch.sum(similarDEEM==1)
                npairssupervised_dissimilar += torch.sum(similarDEEM==0)
                
                writer.add_scalar("Visu/loss_ML_DEEM", lossConst_ML_deem, iter)
                writer.add_scalar("Visu/loss_CNL_DEEM", lossConst_CNL_deem, iter)
                writer.add_scalar("Visu/plijSij_ML_DEEM", plijSij_ML_deem, iter)
                writer.add_scalar("Visu/plijSijbarre_ML_DEEM", plijSijbarre_ML_deem, iter)
                writer.add_scalar("Visu/plijSij_CNL_DEEM", plijSij_CNL_deem, iter)
                writer.add_scalar("Visu/plijSijbarre_CNL_DEEM", plijSijbarre_CNL_deem, iter)
                
            # ********* PAIRS DML TO LABELS **********
            # another way to take the prior : we use the data marked as labeled in DML providing data are loaded similarly... 
            # we sorted files before so that indices should correspond
            # for these marked data we use the LABEL in the loss as before
            lossDMLprior_pairs2labels = 0.0
            if args['useDMLprior_pairs2labels'] == 1:# and train_loader.dataset.prob_sampleforceDMLprior_pairs > 0.0: 
                # similarDML is true or false if the data has been used in DML for training else -1, true=>idx1[i] & idx2[i] same classes
                # false=>idx1[i] & idx2[i] different classes, -1 unknown
                # in DML we use the prior that two images are similar or not without specifying the labels
                # here we use the labels so it is stronger
                lossPr3, lossPr4 = accountPrior(criterion2, bbas1, bbas2, 
                    labels1, labels2, (similarDML != -1), (similarDML != -1))
                
                wsubloss = 0.1
                lossDMLprior_pairs2labels = wsubloss*lossPr3 + wsubloss*lossPr4
                writer.add_scalar("Losses/loss_prior_frDML", lossDMLprior_pairs2labels, iter)
                npairssupervised += torch.sum(similarDML!=-1)
                npairssupervised_similar += torch.sum(similarDML==1)
                npairssupervised_dissimilar += torch.sum(similarDML==0)

            # ********* PAIRS **********
            # another way to take the prior as in EVNN
            # here we use the constraints must link / cannot link, not the labels
            lossDMLprior_pairs = 0.0            
            if args['useDMLprior_pairs'] == 1: # and train_loader.dataset.prob_sampleforceDMLprior_pairs > 0.0:
                # Call the method of Thierry Denoeux et al.
                #lossConst_ML, lossConst_CNL = accountPrior_constraints_TD(bbas1, bbas2, outputs, similarDML)
                #lossConst_ML, lossConst_CNL = accountPrior_constraints_TD_v2(outputs, similarDML)
                lossConst_ML, lossConst_CNL, plijSij_ML, plijSij_CNL, plijSijbarre_ML, plijSijbarre_CNL = \
                    accountPrior_constraints_TD(bbas1, bbas2, outputs, similarDML)
                #lossTDML = lossConst_ML
                #lossTDCNL = lossConst_CNL 
               
                lossDMLprior_pairs = 0.1*lossConst_ML + 0.1*lossConst_CNL
                
                #writer.add_scalar("Losses/loss_ML_v2_frDML", lossConst_ML, iter)
                #writer.add_scalar("Losses/loss_CNL_v2_frDML", lossConst_CNL, iter)
                npairssupervised += torch.sum(similarDML!=-1)
                npairssupervised_similar += torch.sum(similarDML==1)
                npairssupervised_dissimilar += torch.sum(similarDML==0)
                
                # retourne également les pl pour visualisation : par ex tracer la moyenne des pl sur les batchs
                writer.add_scalar("Visu/loss_ML_frDML", lossConst_ML, iter)
                writer.add_scalar("Visu/loss_CNL_frDML", lossConst_CNL, iter)
                writer.add_scalar("Visu/plijSij_ML_frDML", plijSij_ML, iter)
                writer.add_scalar("Visu/plijSijbarre_ML_frDML", plijSijbarre_ML, iter)
                writer.add_scalar("Visu/plijSij_CNL_frDML", plijSij_CNL, iter)
                writer.add_scalar("Visu/plijSijbarre_CNL_frDML", plijSijbarre_CNL, iter)
                
        
        # perform backprop
        total_loss = loss_dijkij + lossprior_labels_deem + lossprior_pairs_deem + lossDMLprior_pairs2labels + lossDMLprior_pairs
        writer.add_scalar("Losses/Loss_tot", total_loss, iter)
        
        # Scale and backpropagate the loss
        scaler.scale(total_loss).backward()
        scaler.step(optimizer) #Applies scaled gradients to update model parameters.
        scaler.update()

        # Compute gradient norm from existing gradients
        total_grad_norm = torch.sqrt(sum(p.grad.norm(2)**2 for p in model.parameters() if p.grad is not None))
        writer.add_scalar('Gradients/totalGrad', total_grad_norm, iter) 
        
        # Check gradient norm for early stopping
        if early_stopping_gradnorm.update(total_grad_norm):
            print(f"Early stopping at epoch {epoch} (Gradient norm: {total_grad_norm:.6f})")
            stopcrit = True
            #break
                
        #optimizer_alpha.step()
        #writer.add_scalar('Gamma/Alpha', alpha, iter) 
        writer.add_scalar('Gamma/Gamma', gam, iter) 
                
        # Scheduler step
        if iter <= nitermax:
            scheduler.step() # Adjusts the learning rate based on the current epoch or iteration.
            
        elif nitermax < iter <= nitermax + niterlastphase:
            # Keep scheduler state; no action needed
            if batch_idx % args['log_interval'] == 0:
                print('NO SCHEDULER - CONSTANT')
                print(scheduler.get_last_lr())
        
        # Log learning rate
        current_lr = scheduler.get_last_lr()
        writer.add_scalar("LR", current_lr[0], iter)
        
        # Validation         
        if batch_idx % touslesvalidation == 0:
            
            avg_loss = total_loss.detach().item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}, \tLR: {}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), avg_loss, scheduler.get_last_lr()))
            print(f"#iter/itermax: {iter} / {nitermax} (+nitlast = {niterlastphase})" )            
            
            if val_loader != []:
                print('VALIDATION DATA')
                nbmaxvalidations = 200000 # nb of pairs to estimate the loss, certainly that images will be redundant so the larger the better
                loss_val_history, aRI_val, aNMI_val, \
                    acc_val, accpm1_val, loss_val = \
                    validate(args, model, criterion1, criterion2, pdist, device, val_loader, train_loader, \
                        d0, nbmaxvalidations, loss_val_history)
                print('\nVALID set aRI: {:.4f}, aNMI: {:.4f}, ACC: {:.4f}, ACCpm1: {:.4f}, loss: {:.4f}\n'.format(aRI_val, aNMI_val, acc_val, accpm1_val, loss_val))
                writer.add_scalar("Losses/Loss_val", loss_val, iter)
                writer.add_scalar("perfVal"+"/aRI", aRI_val, iter)
                writer.add_scalar("perfVal"+"/aNMI", aNMI_val, iter)
                writer.add_scalar("perfVal"+"/ACC", acc_val, iter)
                writer.add_scalar("perfVal"+"/ACCpm1", accpm1_val, iter)
                
                criterion_val = 0.5+aRI_val + 0.3*aNMI_val + 0.2*acc_val
                if criterion_val > best_vals['criterion']:
                    best_vals = {
                        "optimizer": optimizer,
                        "scaler": scaler,
                        "model": model,
                        "iteration": iter,
                        "epoch": epoch,
                        "criterion": criterion_val
                    }
                model.train()
        
        # Test
        if batch_idx % touslestest == 0:
            print('TESTING DATA')               
            test(model, device, test_loader, iter, "perfTest")
            model.train()
    
        if False: #iter % save_interval == 0
            model_path = f'EVNN_model_iter_{iter}.pth'
            model_path = dest_dir + '/' + model_path
            scripted_model = torch.jit.script(model)
            scripted_model.save(model_path)
            print(f"Model saved at iteration {iter} to {model_path}")
            checkpoint = {
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                'epoch': epoch}
            torch.save(checkpoint, dest_dir+'/'+ f'optimstate_iter_{iter}.pth')
            
    if args['useDMLprior_pairs']==1 or args['useDMLprior_pairs2labels']==1:
        s1 = args['batch_size']*len(train_loader)
        s2 = s1*args['prob_sampleforceDMLprior_pairs']
        s3 = args['prob_sampleforceDMLprior_pairs']
        mb = args['batch_size']
        print(f'I used {npairssupervised} pairs in total in this epoch over {s1} data in the train loader (MB={mb})')
        print(f'with sample prob={s3} i expect {s2}*probDMLprior pairs at maximum')
        # MNIST train a 48000 points : Si DML entrainé avec 50% des données => 24000 points labelisés
        # Si on force le tirage de p paires labelisées alors en moyenne p*24000
        print(f'\t{npairssupervised_similar} pairs similar')
        print(f'\t{npairssupervised_dissimilar} pairs dissimilar')
       
        
    return stopcrit, iter, percentile_estimator, loss_val_history, early_stopping_gradnorm, optimizer, scaler, model, best_vals 



##############################################################################
# VALIDATION MODEL
##############################################################################

def validate(args, model, criterion1, criterion2, pdist, device, val_loader, 
             train_loader, d0, nbmaxvalidations, loss_val_history):
    
    model.eval()
    val_loader.dataset.model_DML.eval()
    prob = val_loader.dataset.probabilityUsePrior
    val_loader.dataset.probabilityUsePrior = 0.0 # force random pair otherwise supervised and therefore the perf is much better
    predictions = []
    truelabels = []    
    n = 0 # count nb of images read
    loss_tot = torch.tensor(0.0)
    
    with torch.no_grad():
        
        for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            
            images_1, images_2, labels1, labels2, indexes_prior1, indexes_prior2, similarDEEM, similarDML, precompembedding_img1, precompembedding_img2 = data
            
            
            # Compute embedding with DML or use precomputed embeddings
            if args['training_mode'] == "DML": 
            
                if not hasattr(train_loader.dataset, 'model_DML'): 
                    raise AttributeError('Must have a DML model')        
                if train_loader.dataset.transform_DML is None:
                    raise ValueError("The DML transform must be provided (even if the size does not change!)")
                    
                img1_DML = train_loader.dataset.transform_DML(images_1).to(device)
                img2_DML = train_loader.dataset.transform_DML(images_2).to(device)                                   
            
                with torch.no_grad():
                    embedding_i = train_loader.dataset.model_DML(img1_DML)
                    embedding_j = train_loader.dataset.model_DML(img2_DML)
                    
            elif args['training_mode'] == "PRECOMPUTED_EMBEDDINGS":
                
                embedding_i = precompembedding_img1.to(device)
                embedding_j = precompembedding_img2.to(device)
               
            if args['dist_normalisation'] == 'exp':
                ######################
                # NONLINEAR SCALING 1-exp(-gamma*dij)
                ######################
                    
                targets = pdist(embedding_i, embedding_j)  
                                       
                gam = -torch.log(torch.tensor(0.05)) / (d0)
                
                targets = 1.0 - torch.exp(-gam*(targets))  
                    
            # transfer to device and apply EVNN
            images_1, images_2, targets = images_1.to(device, non_blocking=True), images_2.to(device, non_blocking=True), targets.to(device, non_blocking=True)        
            
            # model output
            outputs, bbas1, bbas2 = model(images_1, images_2)
            
            preds = bbas2plcontour(bbas1)                
            _, predCluster = torch.max(preds, dim=1)        
            predictions.append(predCluster.detach().cpu().numpy())            
            truelabels.append(np.transpose(labels1).detach().cpu().numpy())
            
            preds = bbas2plcontour(bbas2)                
            _, predCluster = torch.max(preds, dim=1)
            predictions.append(predCluster.detach().cpu().numpy())            
            truelabels.append(np.transpose(labels2).detach().cpu().numpy())
                            
            # compute loss
            loss = criterion1(outputs, targets)
            
            # accumulate                    
            loss_tot = loss_tot + loss  
            
            n = n + images_1.shape[0] + images_2.shape[0]
            if n > nbmaxvalidations: 
                print(f"Validated {n} stopped")
                break

    # average the loss
    loss_tot = loss_tot / torch.tensor(n)
    loss_val_history.update(loss_tot)
                         
    # Accuracy
    predictions = np.concatenate(predictions)
    truelabels = np.concatenate(truelabels)        
    aRI = adjusted_rand_score(truelabels, predictions)   
    aNMI = normalized_mutual_info_score(truelabels, predictions)   
    acc = cluster_accuracy(truelabels, predictions)        
    accpm1 =  (np.sum(predictions == truelabels) + np.sum((predictions-1) == truelabels) + np.sum((predictions+1) == truelabels))/len(predictions)
        
    # reload the prob
    val_loader.dataset.probabilityUsePrior = prob
    
    return loss_val_history, aRI, aNMI, acc, accpm1, loss_tot

##############################################################################
# TEST MODEL
##############################################################################

def test(model, device, test_loader, iter, tagWriterTB, early_stopping=None):
    
    model.eval()
    
    predictions = []
    truelabels = []
    
    with torch.no_grad():
    
        for batch_idx, data in enumerate(test_loader):            
            images, labels = data
            images = images.to(device)
            _, bbas, _ = model(images, images)
            #assert bbas.shape[0] == images.shape[0], f"Mismatch between bbas ({bbas.shape[0]}) and batch size ({images.shape[0]})"            
            preds = bbas2plcontour(bbas)                
            _, predCluster = torch.max(preds, dim=1)
            predictions.append(predCluster.detach().cpu().numpy())            
            truelabels.append(np.transpose(labels).detach().cpu().numpy())
        
        predictions = np.concatenate(predictions)
        truelabels = np.concatenate(truelabels)        
        aRI = adjusted_rand_score(truelabels, predictions)   
        aNMI = normalized_mutual_info_score(truelabels, predictions)   
        cm = contingency_matrix(truelabels, predictions)
        paircm = pair_confusion_matrix(truelabels, predictions) # https://scikit-learn.org/stable/modules/clustering.html#pair-confusion-matrix
        acc = cluster_accuracy(truelabels, predictions)
        # accu at plus or minus works with supervision otherwise non sense !!!
        accpm1 =  (np.sum(predictions == truelabels) + np.sum((predictions-1) == truelabels) + np.sum((predictions+1) == truelabels))/len(predictions)
       
        writer.add_scalar(tagWriterTB+"/aRI", aRI, iter)
        writer.add_scalar(tagWriterTB+"/aNMI", aNMI, iter)
        writer.add_scalar(tagWriterTB+"/ACC", acc, iter)
        writer.add_scalar(tagWriterTB+"/ACCpm1", accpm1, iter)
        #writer.add_image(tagWriterTB+"/matrix_image", torch.tensor(255*(cm/cm.max()), dtype=torch.uint8).unsqueeze(0), global_step=iter)
        
        if early_stopping is not None:
            early_stopping(aNMI, model)      # if val_acc decrease "patience" times then stop
            if early_stopping.early_stop:    # check
                print("Early stopping (aNMI)")                
            
    print('\nTest set aRI ({:d} data): {:.4f}, aNMI: {:.4f}, ACC: {:.4f}, ACCpm1: {:.4f}\n'.format(len(predictions),aRI, aNMI, acc, accpm1))
    print('Contingency matrix:')
    print(cm)
    #print('\nPair confusion matrix:') 
    #print(paircm), print('\n')
    
    model.train()
    
    return aRI, aNMI, cm, paircm, acc, accpm1, early_stopping

    
##############################################################################
# THE MAIN
##############################################################################

def main():

    # load training data
    train_loader, val_loader, test_loader, _, d090 = data_loader_DML(args['dataset'])    
    # load model 
    model = model_loader()
    # load optimiser
    optimizer, scheduler, nitermax, niterlastphase = opt_loader(model)
    
    # load a pretrained model
    path2pretrainedmodel = args['model_pretrained']
    if not(path2pretrainedmodel == ""):
        print(f'Pretrained model {path2pretrainedmodel}...')
        scripted_model = torch.jit.load(path2pretrainedmodel)
        model.load_state_dict(scripted_model.state_dict())  # Load weights
        model = model.to(device)
        # possibility also to load the optimiser state
        #checkpoint = torch.load('test_tsne_0.9priorpairs_DEEM_repo/optimstate_final.pth')
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # we look at variables in a sliding window
    percentile_estimator = SlidingWindow(window_size=100000)
    loss_val_history = EarlyStopping(window_size=500, patience=5, tolerance_ratio=0.05, min_val=0.05)
    early_stopping_gradnorm = EarlyStopping(window_size=500, patience=5, tolerance_ratio=0.05, min_val=0.05)
    best_vals = {"optimizer": None, "scaler": None, "model": None, "iteration": 0, "epoch": 0, "criterion": 0.0}
    
    #alpha = nn.Parameter(torch.tensor(0.0))
    #optimizer_alpha = torch.optim.Adam([alpha], lr=1e-3)
    
    # Run
    print('Go...')
    for epoch in range(1, args['num_epochs'] + 1):
        
        stopcrit, iter, percentile_estimator, loss_val_history, early_stopping_gradnorm, \
            optimizer, scaler, model, best_vals = \
            train(args, model, device, train_loader, val_loader, test_loader, 
            optimizer, scheduler, epoch, nitermax, niterlastphase, 
            d090, percentile_estimator, early_stopping_gradnorm, loss_val_history, best_vals)
        
        if stopcrit == True:
            print('Stop crit was true')
            break
    
    print('Final test...')
    test(model, device, test_loader, iter, "perfTest")
    writer.close()
    
    print('Saving...')    
    model_path = f'EVNN_model_final.pth'
    model_path = dest_dir + '/' + model_path
    scripted_model = torch.jit.script(model)
    scripted_model.save(model_path)
    print(f"Model saved to {model_path}")
    checkpoint = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),  # Save scaler state
        'epoch': epoch,
        'iter': iter}
    torch.save(checkpoint, dest_dir+'/'+ f'optimstate_final.pth')    
    
    if best_vals['model'] is not None: # case where validation set was used
        model_path = f'EVNN_model_bestval.pth'
        model_path = dest_dir + '/' + model_path
        scripted_model = torch.jit.script(best_vals['model'])
        scripted_model.save(model_path)
        print(f"Best model saved to {model_path}")
        checkpoint = {
            'optimizer_state_dict': best_vals['optimizer'].state_dict(),
            'scaler_state_dict': best_vals['scaler'].state_dict(),  # Save scaler state
            'epoch': best_vals['epoch'], 
            'iter': best_vals['iteration'],
            'perf': best_vals['criterion']}
        torch.save(checkpoint, dest_dir+'/'+ f'optimstate_bestmoddel.pth')
    
    
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('time')
    stats.print_stats(10)  # Top 10 functions
