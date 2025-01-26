Guiziou, L., Ramasso, E., Thibaud, S., & Denneulin, S. (2024, August). Deep Evidential Clustering of Images. In International Conference on Belief Functions (pp. 3-12). Cham: Springer Nature Switzerland.


MNIST_MAIN corresponds to the main file.
Some parameters are set by the user:

numClasses = 10;        % for MNIST, 10 classes are used. It should be adapted to any dataset.

si=5000;                % Number of training images. In this case, "si" limit is 10000, corresponding to the entire MNIST-test. 
			% Because we test on 60,000 images, we invert test and train datasets for MNIST.

priorIdx = []; 		% Array of supervised images. Use index of image, ex: [1,5,6,53] or linspace(1,500,500).
			% Works even if supervised boolean is set to false !

% Determining Dij
supervised = false;	% Only for Dij. If set to true, Dij are a combination of distances and determined dissimilarities from true classes.
HS=2; 			% Quantity of prior in the combination of Dij (supervised only). See HSL array.
st=2; 			% Noise in perfect Dij (supervised or preclustering). See sig array.
preclustering = true; 	% Only on unsupervised mode. If false, Dij stays determined thanks to distances from the t-SNE embdedding "Y".

miniBatchSize = 128; % 64 1024;
numEpochs = 150; % 50
iterMaxUser = 7000;