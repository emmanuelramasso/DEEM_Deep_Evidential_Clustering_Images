import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from scipy.io import loadmat
from EVNN_config import NUMCLASSES, SIZE_IMAGES, network, coloredinput, device, dataset
from EVNN_utils import get_Cmatrix_and_forDecision
       

##############################################################################
# Load the model
##############################################################################

print(f'SIZE of assumed INPUT IMAGES {SIZE_IMAGES}')
def model_loader():
    match network:
        case 'simple1':
            if dataset=="mnist":
                model = CustomCNN(device=device,numChannels=1).to(device)
            elif dataset=="orion":
                model = CustomCNN(device=device,numChannels=3).to(device)            
        case 'resnet18':
            model = SiameseNetwork(device,coloredinput).to(device)
    
    #model.to(device=device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the network: ', total_params)
    return model 

##############################################################################
# DEFINE DIFFERENT NETWORKS
##############################################################################

class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """
    def __init__(self, device,coloredinput=False):
        super(SiameseNetwork, self).__init__()
        self.device = device
        self.Cmatrix, self.forDecision = get_Cmatrix_and_forDecision(NUMCLASSES)
        self.noutputs = self.Cmatrix.shape[0]
        
        # Get ResNet model
        self.resnet = torchvision.models.resnet18(pretrained=False)

        # Overwrite the first conv layer to read MNIST or ORION images -- TO BE DONE IMPROVE FLEXIBILITY IN IMAGE SIZE
        if coloredinput==False:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            print(coloredinput)
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # Remove the last layer of ResNet18
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # Add a linear layer to represent all masses on singletons + doubt + subset of 2 + conflict => 57
        self.fc = nn.Linear(self.fc_in_features, self.noutputs)

        self.softmax = nn.Softmax(dim=1) # vector inputs

        # Initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def getconflict(self, bba1, bba2):
        # bba1: [batch_size, nbFocsets]
        # Cmatrix: [nbFocsets, nbFocsets] (assuming it was transposed correctly)
        
        # Compute mtC as the matrix multiplication of bba1 and Cmatrix
        mtC = torch.matmul(bba1, self.Cmatrix)  # [batch_size, nbFocsets]
        
        # Compute conflict Kij
        Kij = torch.sum(mtC * bba2, dim=1)  # [batch_size]
        
        return Kij
    
    def forward(self, input1, input2):
        
        # Get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Pass through linear layer
        output1 = self.fc(output1)
        output2 = self.fc(output2)

        # Pass through softmax
        bba1 = self.softmax(output1)
        bba2 = self.softmax(output2)
        
        # Combine both BBAs
        conflict = self.getconflict(bba1, bba2)

        return conflict, bba1, bba2


##############################################################################
# A SIMPLER NET

def conv_output_size(input_size, kernel_size, stride, padding):
    output_size = ((input_size + 2 * padding - kernel_size) // stride) + 1
    return output_size

class CustomCNN(nn.Module):
    def __init__(self, device, numChannels):
        super(CustomCNN, self).__init__()

        self.device = device
        self.Cmatrix, self.forDecision = get_Cmatrix_and_forDecision(NUMCLASSES)        
        self.noutputs = self.Cmatrix.shape[0]
        
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=12, kernel_size=4, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(12)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=4, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        
        # Calculate the output size of the last convolutional layer
        self.last_conv_size = self._calculate_last_conv_size()

        # Define the fully connected layer
        #self.fc = nn.Linear(self.last_conv_size*84, self.noutputs)
        self.fc = nn.Linear(self.last_conv_size, self.noutputs)

        self.softmax = nn.Softmax(dim=1) # vector inputs

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Apply Xavier Normal initialization to all convolutional and linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def getconflict(self, bba1, bba2):
        # bba1: [batch_size, nbFocsets]
        # Cmatrix: [nbFocsets, nbFocsets] (assuming it was transposed correctly)
        
        # Compute mtC as the matrix multiplication of bba1 and Cmatrix
        mtC = torch.matmul(bba1, self.Cmatrix)  # [batch_size, nbFocsets]
        
        # Compute conflict Kij
        Kij = torch.sum(mtC * bba2, dim=1)  # [batch_size]
        
        return Kij

    def forward_once(self, x):
        # Forward pass through convolutional layers
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        # Flatten the output from the last convolutional layer
        x = torch.flatten(x, 1)  # Alternatively, x = x.view(x.size(0), -1)
         # Pass through linear layer
        x = self.fc(x)
        return x
    
    def forward(self, input1, input2):        
        # Get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Pass through softmax
        bba1 = self.softmax(output1)
        bba2 = self.softmax(output2)
        
        # Combine both BBAs
        conflict = self.getconflict(bba1, bba2)

        return conflict, bba1, bba2

    def _calculate_last_conv_size(self):
        # Assuming input image size is DxD
        #print(f'SIZE_IMAGES is {SIZE_IMAGES}')
        dummy_input = torch.zeros(1, SIZE_IMAGES[0], SIZE_IMAGES[1], SIZE_IMAGES[2])#.to(self.device)
        x = self.conv1(dummy_input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        return x.numel()

