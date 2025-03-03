# python3 EVNN.py --jsonfile config000001.json 

import torch
import torch.optim as optim

from EVNN_config import args
from torch.optim.lr_scheduler import _LRScheduler

##############################################################################
# Calculate expected values for scheduler 1cycle particularly
def sanity_check_1cycle(training_dataset_size):
    # check is total step is coherent with calculated value
    
    batch_size = args['batch_size']
    steps_per_epoch = training_dataset_size // batch_size # iterations
    print(f'Expected #iterations (steps) per epoch given #dataset {training_dataset_size} and batch size {batch_size} is {steps_per_epoch}')
    
    expected_nitermax_given_epochs = steps_per_epoch * args['num_epochs']
    print(f'Expected #total iterations given #epochs {expected_nitermax_given_epochs}')
    provided_nitermax = args['nitermax']
    print(f'Provided #total iterations {provided_nitermax}')
        
    # Check nitermax
    #if provided_nitermax != expected_nitermax_given_epochs:
    #    raise ValueError(f"nitermax is not coherent. Correct the json file.")

    print('Sanity check passed for 1cycle scheduler')
    
##############################################################################
def opt_loader(model):
    
    # --------------------------------------
    # Optimizer
    # --------------------------------------
    if args['optimizer']=='adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['base_learning_rate'])
    elif args['optimizer']=='sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args['base_learning_rate'])
    elif args['optimizer']=='rmsprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args['base_learning_rate'])
    elif args['optimizer']=='adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args['base_learning_rate'], weight_decay=args['adamw_weight_decay'])#, betas=(0.9, 0.999))
    elif args['optimizer']=='radam':
        optimizer = optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['base_learning_rate'], weight_decay=args['adamw_weight_decay'])#, betas=(0.9, 0.999))
    else:
        raise "Scheduler unknown"

    niterlastphase = args['niterlastphase'] #500 for a third phase maintaining the LR
    nitermax = args['nitermax']
        
    if args['scheduler']=='CyclicLR':            
            
        print('CYCLIC LR')            
        step_size_up = args['step_size_up_CyclicLR'] #2000
        step_size_down = args['step_size_down_CyclicLR'] #4000
        assert(step_size_up+step_size_down+niterlastphase<=nitermax)
            
        print('COSINE+WR')         
        #base_learning_rate is used as eta_max (https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)   
        #and eta_min must be < eta_max
        assert(args['eta_min'] < args['base_learning_rate'])
        print('Base LR (eta_max) : ', args['base_learning_rate'])
        print('Min LR (eta_min) : ', args['eta_min'])            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2000, T_mult=1, eta_min=args['eta_min'], last_epoch=-1)
        
    elif args['scheduler']=='StepLR':  
            
        print('STEP LR')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size_StepLR'], gamma=args['gamma_StepLR'])
            
    elif args['scheduler']=='OneCycleLR':  
            
        print('ONECYCLE LR')        
        anneal_strategy = 'cos' #'linear'
        div_factor = 25.0   
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args['base_learning_rate'], div_factor=div_factor,
            total_steps = nitermax, pct_start=0.3, anneal_strategy=anneal_strategy, three_phase=args['3phases'])
            
    elif args['scheduler']=='ConstantLR':
        if type(args['nbepoch_apply_gamma']) is str and args['nbepoch_apply_gamma'] == "inf":
            total_iters = float('inf')
        else:
            total_iters = args['nbepoch_apply_gamma']
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=args['gamma_constLR'], total_iters=total_iters)

    elif args['scheduler'] == 'staircase':

        # Define scheduler milestones and learning rates        
        # Parameters
        start_lr = 0.0001
        increment = 0.00025
        start_iter = 1500
        step_iters = 1000
        peak_lr = 0.0025
        end_iter = 150000
        min_lr_when_drecrement = start_lr
        milestones, learning_rates = generate_milestones_and_lrs(
            start_lr, increment, start_iter, step_iters, peak_lr, end_iter, min_lr_when_drecrement)

        print("Milestones:", milestones)
        print("Learning Rates:", learning_rates)

        scheduler = IterationBasedScheduler(optimizer, milestones, learning_rates)

    else:
        raise "Unknown scheduler"    
    
    return optimizer, scheduler, nitermax, niterlastphase



class IterationBasedScheduler(_LRScheduler):
    def __init__(self, optimizer, milestones, lrs, last_epoch=-1):
        """
        Custom scheduler that changes learning rates at specific iteration milestones.
        
        Args:
            optimizer: PyTorch optimizer.
            milestones: List of iteration milestones where LR changes.
            lrs: List of learning rates corresponding to each milestone.
            last_epoch: The index of last epoch. Default: -1.
        """
        assert len(milestones) == len(lrs) - 1, "milestones length must be len(lrs) - 1"
        self.milestones = milestones
        self.lrs = lrs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        iteration = self.last_epoch
        for i, milestone in enumerate(self.milestones):
            if iteration < milestone:
                return [self.lrs[i] for _ in self.base_lrs]
        return [self.lrs[-1] for _ in self.base_lrs]


def generate_milestones_and_lrs(start_lr, increment, start_iter, step_iters, peak_lr, end_iter, min_lr_when_drecrement):
    """
    Automatically generate milestones and learning rates.
    
    Args:
        start_lr (float): Initial learning rate.
        increment (float): Amount to increment/decrement the LR at each step.
        start_iter (int): Starting iteration for LR changes.
        step_iters (int): Number of iterations between LR changes.
        peak_lr (float): Maximum learning rate to reach.
        end_iter (int): Final iteration where learning rate changes stop.
        min_lr_when_drecrement (float): Ensures LR doesn't go below this threshold
        
    Returns:
        milestones (list): List of iteration milestones.
        learning_rates (list): Corresponding learning rates for each milestone.
    """
    milestones = []
    learning_rates = []
    
    # Increasing phase
    current_iter = start_iter
    current_lr = start_lr
    
    while current_lr <= peak_lr and current_iter < end_iter:
        milestones.append(current_iter)
        learning_rates.append(current_lr)
        current_lr += increment
        current_iter += step_iters

    # Decreasing phase
    while current_lr > 0 and current_iter < end_iter:
        current_lr -= increment
        if current_lr < min_lr_when_drecrement:
            current_lr = min_lr_when_drecrement  # Ensure LR doesn't go below min_lr_when_drecrement
        milestones.append(current_iter)
        learning_rates.append(current_lr)
        current_iter += step_iters
    
    return milestones, learning_rates
