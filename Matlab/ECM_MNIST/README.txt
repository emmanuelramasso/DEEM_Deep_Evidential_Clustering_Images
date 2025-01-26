ECM is a Matlab function available at 
https://www.hds.utc.fr/~tdenoeux/dokuwiki/en/software/ecm
and comming from 
M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy c-means algorithm. Pattern Recognition, Vol. 41, pages 1384– 1397, 2008.

This function can be executed in Matlab using the following steps:

clear all
load('MNIST_5k_data.mat')        % Loading the dataset
K=10;                             % Number of clusters
Y = zscore(Y); % normalise

[m,g,F,pl,BetP,J,N] = ECM(Y,K,1); % Main function, default param with pairs
[~,res]=max(pl,[],2);                 % res stores the assigned labels
valid_RandIndex(res,minibasetraining.labels)

[m,g,F,pl,BetP,J,N] = ECM(Y,K,1,2,10,1); % with FCM
[~,res]=max(pl,[],2);                 % res stores the assigned labels
valid_RandIndex(res,minibasetraining.labels)

MNIST_5k_data.mat contains:
- minibasetraining                % A cell array containing image data and labels from MNIST-test
	- minibasetraining.images % Corresponding images
	- minibasetraining.labels % Corresponding true classes
- Y                               % A t-SNE embedding of the data