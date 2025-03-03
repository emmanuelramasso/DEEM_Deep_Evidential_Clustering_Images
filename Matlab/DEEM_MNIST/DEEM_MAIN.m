% VERSION 1.0 01/26/2025
% Loïc Guiziou and Emmanuel Ramasso 
% 
% This file shows how were implemented the first tests for our article 
% "DEEM: A Novel Approach to Semi-Supervised and Unsupervised Image " + ...
% Clustering under Uncertainty using Belief Functions and Convolutional
% Neural Networks"
% 
% Tests concern DEEM clustering with various amounts of prior
% In this example it runs with prior and with preclustering
% DEEM is trained and then used to infer on a test set. 
% To use prior set "supervised" to true. Data are from MNIST.
%
% The comments are to be improved (to be updated around March 2025)
% together with a Python version. In this version you can already manage 
% many options but prior on pairs is not managed, nor DML which were
% coded in the Python.
% 
%

clear all
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultLegendFontSize', 25);
set(groot, 'defaultLegendFontName', 'Times');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultAxesFontSize', 25);
set(groot, 'defaultAxesFontName', 'Times');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultAxesFontSize', 25);
set(groot, 'defaultAxesFontName', 'Times');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultLegendFontSize', 25);
set(groot, 'defaultLegendFontName', 'Times');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultTextFontSize', 25);
set(groot, 'defaultTextFontName', 'Times');
gpuDevice(1)

%% Settings
numClasses = 10;
si=5000; % Number of training images
priorIdx = []; % Array of supervised images

supervised = false; % Determining Dij
st=2; % Noise in supervised Dij. See sig array
HS=2; % Quantity of prior. See HSL array
preclustering = true; % Only on unsupervised mode

miniBatchSize = 128; % 64 1024;
numEpochs = 150; % 50
iterMaxUser = 7000;


%% Load and prepare data
disp('Loading data...')

load('mnist.mat');

minibasetraining.images=test.images(:,:,1:si);
minibasetraining.labels=test.labels(1:si)+1;
test.images=training.images;
test.labels=training.labels+1;  % Evaluation will not tolerate class 0

if 1
    disp('Creating an embedding...')
    imagesVect = reshape(minibasetraining.images,28*28,[]);
    Y=tsne(imagesVect','Distance','cosine','NumDimensions',2);
    [clustersKMean,Cent] = kmeans(Y,10,'Replicates',10);
    [ARIk,RI,MI,HI,C] = valid_RandIndex(clustersKMean,minibasetraining.labels);
    NMIk = nmi(clustersKMean,double(minibasetraining.labels));
    ACCk=cluster_acc(minibasetraining.labels,clustersKMean);
    figure, gscatter(Y(:,1),Y(:,2),minibasetraining.labels)

    %Dij=pdist(Y);
    %Dij=Dij./max(Dij);
end

if 0
    Dij1=pdist(imagesVect','euclidean');
    Dij2=pdist(imagesVect','squaredeuclidean');
    Dij3=pdist(imagesVect','seuclidean');
    Dij4=pdist(imagesVect','mahalanobis');
    Dij5=pdist(imagesVect','cityblock');
    Dij6=pdist(imagesVect','minkowski');
    Dij7=pdist(imagesVect','chebychev');
    Dij8=pdist(imagesVect','cosine');
    Dij9=pdist(imagesVect','correlation');
    Dij10=pdist(imagesVect','hamming');
    Dij11=pdist(imagesVect','jaccard');
    Dij12=pdist(imagesVect','spearman');

    Dij=Dij1;
end

% [~, score, ~, ~, explained] = pca(Y);
% f = find(cumsum(explained)>99.99,1,'first');
% Y = score(:,1:f); %supression des descripteurs moins significatifs


sig=linspace(0,1,11);


n = length(minibasetraining.labels);
smallIijidx = itril(n,-1)';
smallIijidx = smallIijidx-1; % trick
col = floor(smallIijidx/n) + 1; % correspondance colonne
lig = mod(smallIijidx,n) + 1; % correspondance ligne

if supervised
    Dij2=zeros(si);
    for i=1:si
        for j=1:i
            if minibasetraining.labels(i)==minibasetraining.labels(j)
                Dij2(i,j)=0;
                Dij2(j,i)=0;
            else
                Dij2(i,j)=1;
                Dij2(j,i)=1;
            end
        end
    end

    for i=1:si
        for j=1:i
            r=rand(1);
            if Dij2(i,j)==0
                Dij2(i,j)=Dij2(i,j)+r*sig(st);
            else
                Dij2(i,j)=Dij2(i,j)-r*sig(st);
            end
            if i==j
                Dij2(i,j)=0;
            else
                Dij2(j,i)=Dij2(i,j);
            end
        end
    end
    % % % Dij=Dijm.*(pr/10)+Dij2.*(1-pr/10);
    % % figure, image(Dij*255)
    Dij2 = squareform(Dij2,'tovector');

    HSL=[2,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.025,0.005]./2;

    nDij = length(Dij);
    n_elements = round(HSL(HS) * nDij);
    list=[ones(1,n_elements),zeros(1,nDij-n_elements)];
    list=list(randperm(nDij));

    for d=1:nDij
        if list(d)==1
            Dij(d)=Dij2(d);
        end
    end

else
    %Dij=pdist(Y);
    %Dij=Dij./max(Dij);

    if preclustering

        % gmm = fitgmdist(data, 10);
        % clusterIdx = cluster(gmm, data);

        clust=linkage(Y,'ward');
        clusterIdx = cluster(clust, 'maxclust', 10);

        [ARIe,RIe,MIe,HIe,Ce] = valid_RandIndex(clusterIdx,minibasetraining.labels);
        NMIe = nmi(clusterIdx,double(minibasetraining.labels));
        ACCe=cluster_acc(minibasetraining.labels,clusterIdx);

        Dij=zeros(si);
        for i=1:si
            for j=1:i
                if clusterIdx(i)==clusterIdx(j)
                    Dij(i,j)=0;
                    Dij(j,i)=0;
                else
                    Dij(i,j)=1;
                    Dij(j,i)=1;
                end
            end
        end
        for i=1:si
            for j=1:i
                r=rand(1);
                if Dij(i,j)==0
                    Dij(i,j)=Dij(i,j)+r*sig(st);
                else
                    Dij(i,j)=Dij(i,j)-r*sig(st);
                end
                if i==j
                    Dij(i,j)=0;
                else
                    Dij(j,i)=Dij(i,j);
                end
            end
        end
        Dij = squareform(Dij,'tovector');

    end

end


images_ds = [];

trainingFeatures=[];

disp('PCA...'); tic
[coeff, score, latent, tsquared, explained] = pca(trainingFeatures);
disp(toc)
f=find(cumsum(explained)>99,1,'first');
trainingFeaturesHOG_PCA = score(:,1:f);
clear trainingFeatures coeff latent tsquared explained score


eval='ARI';
optionsEVNN = struct(...
    'normalisationInput',           "rescale-zero-one" , ...%rescale-symmetric", ... %rescale-zero-one
    ... %none
    ... %zscore
    ... %zerocenter
    ... %rescale-symmetric
    'imresize',                      [], ...% [224,224], ...
    'usepaires',                     true, ...
    'usedoute',                      true, ...  % if true then use conflit
    'useApriori',                    true, ...
    'selectionDIJthreshold',         @(Dij)quantile(Dij,0.90,'all'), ...
    'initClusters',                  false, ... % IMPLEMENTER D'ABORD GRADIENTS
    'loss_type',                     struct('type','l2loss','param',1), ... % smoothL1Loss mse, e_mae, huber, pseudohuber, logcosh, qreg
    'numClasses',                    [], ...
    'postNormaliseDIJ_ab',           [0,1] ... % [0, 1], ... % 0.2 ou 0.3 (best) pour Wine
    )


% set some variables used later
optionsEVNN.dimX = size(minibasetraining.images,[1 2]);
optionsEVNN.ndata = size(minibasetraining.images,3);
optionsEVNN.numClasses = numClasses;
optionsEVNN.npairesmasses = numClasses*(numClasses-1)/2;

%%
% Now prepare the dissimilarities

% normalise Dij
% mxDij = max(Dij);
% Dij = Dij / mxDij; % normalise
delta0 = optionsEVNN.selectionDIJthreshold(Dij); % plus selectionne le seuil

% gam = -log(0.05)/delta0^2;% SQUARE OR NOT ?? SI EUCLIDEAN YES
% Dij = 1 - exp(-gam*Dij.^2);

% IL Y A UN EFFET DE SEUILLAGE
% SI ON LAISSE 0 et 1 en min et max le modèle converge moins bien
% si on crée une marge c'est bcp mieux
% comme le fait Thierry avec le 1-exp ou bien avec seuil

if size(Dij,1) < 1000000
    clear s
    lab = minibasetraining.labels;
    f11=find(...
        (lab(lig)==1 & lab(col)==1) | ...
        (lab(lig)==2 & lab(col)==2) | ...
        (lab(lig)==3 & lab(col)==3) | ...
        (lab(lig)==4 & lab(col)==4) | ...
        (lab(lig)==5 & lab(col)==5) | ...
        (lab(lig)==6 & lab(col)==6) | ...
        (lab(lig)==7 & lab(col)==7) | ...
        (lab(lig)==8 & lab(col)==8) | ...
        (lab(lig)==9 & lab(col)==9) | ...
        (lab(lig)==10 & lab(col)==10));
    figure('color', 'w')

    histogram(Dij(f11)),s{1}=('Dissimilarities for images in the same class');

    % cherche les labels où les Xi sont de classes differentes
    f11=find(...
        (lab(lig)==1 & lab(col)~=1) | ...
        (lab(lig)==2 & lab(col)~=2) | ...
        (lab(lig)==3 & lab(col)~=3) | ...
        (lab(lig)==4 & lab(col)~=4) | ...
        (lab(lig)==5 & lab(col)~=5) | ...
        (lab(lig)==6 & lab(col)~=6) | ...
        (lab(lig)==7 & lab(col)~=7) | ...
        (lab(lig)==8 & lab(col)~=8) | ...
        (lab(lig)==9 & lab(col)~=9) | ...
        (lab(lig)==10 & lab(col)~=10));
    fdiff = f11;
    hold on, histogram(Dij(f11)),s{2}=('Dissimilarities for images in different classes');
    xlabel('Dissimilarities','Interpreter', 'latex','FontSize',25)
    ylabel('Number of occurrences','Interpreter', 'latex','FontSize',25)
    set(gca, 'TickLabelInterpreter', 'latex');
    ax = gca;
    ax.FontSize=25;
    legend(s,'location','northwest','Interpreter', 'latex','FontSize',25)

    lab = minibasetraining.labels;
    figure, hold on, clear s
    for i=1:10
        f11=find((lab(lig)==i & lab(col)==i));
        histogram(Dij(f11)),s{i}=['Dij images meme classe ' num2str(i)];
    end
    histogram(Dij(fdiff)),s{end+1}=('Dij pour images dans classes differentes');
    legend(s,'location','best')

    clear f11 lab f22 f33 s fdiff
end
logFact = nan; gam = nan;
if (supervised & HS~=1)|(~supervised & ~preclustering)
    %Dij = Dij / max(Dij);
    logFact = 0.05;
    gam = -log(logFact)/delta0;
    Dij = 1 - exp(-gam*Dij);
    % figure,image(squareform(Dij,'tomatrix')*255)
end


assert(~any(Dij>1));

if size(Dij,1) < 1000000
    clear s
    lab = minibasetraining.labels;
    f11=find(...
        (lab(lig)==1 & lab(col)==1) | ...
        (lab(lig)==2 & lab(col)==2) | ...
        (lab(lig)==3 & lab(col)==3) | ...
        (lab(lig)==4 & lab(col)==4) | ...
        (lab(lig)==5 & lab(col)==5) | ...
        (lab(lig)==6 & lab(col)==6) | ...
        (lab(lig)==7 & lab(col)==7) | ...
        (lab(lig)==8 & lab(col)==8) | ...
        (lab(lig)==9 & lab(col)==9) | ...
        (lab(lig)==10 & lab(col)==10));
    figure, histogram(Dij(f11)),s{1}=('Dij pour images dans meme classe');

    % cherche les labels où les Xi sont de classes differentes
    f11=find(...
        (lab(lig)==1 & lab(col)~=1) | ...
        (lab(lig)==2 & lab(col)~=2) | ...
        (lab(lig)==3 & lab(col)~=3) | ...
        (lab(lig)==4 & lab(col)~=4) | ...
        (lab(lig)==5 & lab(col)~=5) | ...
        (lab(lig)==6 & lab(col)~=6) | ...
        (lab(lig)==7 & lab(col)~=7) | ...
        (lab(lig)==8 & lab(col)~=8) | ...
        (lab(lig)==9 & lab(col)~=9) | ...
        (lab(lig)==10 & lab(col)~=10));
    fdiff = f11;
    hold on, histogram(Dij(f11)),s{2}=('Dij images classes differentes');
    legend(s,'location','best')

    lab = minibasetraining.labels;
    figure, hold on, clear s
    for i=1:10
        f11=find((lab(lig)==i & lab(col)==i));
        histogram(Dij(f11)),s{i}=['Dij images meme classe ' num2str(i)];
    end
    %histogram(Dij(fdiff)),s{end+1}=('Dij pour images dans classes differentes');
    legend(s,'location','best')

    clear f11 lab f22 f33 s fdiff
end


Dij_ds = arrayDatastore([Dij;lig;col]');
nDij = length(Dij);

clear Dij lig col

%% Define Network

% The current standard approach for initialization of the weights of neural network
% layers and nodes that use the rectified linear (ReLU) activation function is called "he" initialization.
% Xavier/Glorot Initialization is suitable for layers where the activation function used is Sigmoid.
%
% Options of fullyConnectedLayer to initialise weights and biases
%       'WeightsInitializer'      - The function to initialize the weights,
%                                   specified as 'glorot', 'he', 'orthogonal', 'narrow-normal', 'zeros',
%                                   'ones' or a function handle. The default is 'glorot'.
%       'BiasInitializer'         - The function to initialize the bias, specified as 'narrow-normal',
%                                   'zeros', 'ones' or a function handle. The default is 'zeros'.
weightsInitType = 'he'%'he'; % with relu for deep NN %'narrow-normal';
%weightsInitType = 'narrow-normal';%'glorot'; % for sigmoid or tanh with non deep NN

sizeLastLayer = numClasses + 2*double(optionsEVNN.usedoute==true) + ...
    double(optionsEVNN.usepaires==true)*optionsEVNN.npairesmasses


if 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%% OK %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % marche bien
    layers = [
        imageInputLayer([size(minibasetraining.images,1) size(minibasetraining.images,2) 1],Normalization="none")%optionsEVNN.normalisationInput)
        %rescale-symmetric  %rescale-zero-one
        %none       %zscore         %zerocenter

        convolution2dLayer(4,12,'Padding','same','Stride',2,'WeightsInitializer',weightsInitType)

        batchNormalizationLayer

        %%%%%%maxPooling2dLayer(2,'Stride',2)
        reluLayer

        convolution2dLayer(4,12,'Padding','same','Stride',2,'WeightsInitializer',weightsInitType)

        batchNormalizationLayer

        maxPooling2dLayer(2,'Stride',2)
        reluLayer

        %globalAveragePooling2dLayer
        %globalMaxPooling2dLayer

        fullyConnectedLayer(sizeLastLayer,'WeightsInitializer',weightsInitType)
        ];


end
% % % % % % % % % % % % % % % % % % % % % % % % % %
% % SI UNE BN EST UTILISE ALORS UPDATE DLNET AVEC STATE !!
% % % % % % % % % % % % % % % % % % % % % % % % % %
bnstate = false;
for i=1:length(layers)
    if strcmp(class(layers(i)),'nnet.cnn.layer.BatchNormalizationLayer')
        bnstate = true;
        break;
    end
end

% analyzeNetwork(layers)

%%
% Create a |dlnetwork| object from the layer graph.

dlnet = dlnetwork(layers)
% clear layers
rr=dlnet.Learnables.Value;
pp=0; for i=1:length(rr), pp=pp+length(rr{i}(:)); end
pp % total number of parameters

%%
% also the parametrization of EVNN

% # focal sets
% fs = numClasses + 2*double(optionsEVNN.usedoute==true) + double(optionsEVNN.usepaires==true)*optionsEVNN.npairesmasses;
optionsEVNN.fs = sizeLastLayer;

optionsEVNN.delta0 = delta0;
optionsEVNN.gam = gam;
optionsEVNN.dataformat = 'BC';

npairesreelx = optionsEVNN.ndata*(optionsEVNN.ndata-1)/2;
pairCoding = [];
if optionsEVNN.usepaires
    pairCoding = zeros(optionsEVNN.npairesmasses, 2);
    k = 1;
    for i=1:numClasses
        for j=i+1:numClasses
            pairCoding(k,:) = [i,j];
            k = k+1;
        end
    end
    optionsEVNN.pairCoding = pairCoding;
end

%% Define Model Gradients Function
% Create the function |modelGradients|, listed at the end of the example, that
% takes a |dlnetwork| object, a mini-batch of input data with corresponding labels
% and returns the gradients of the loss with respect to the learnable parameters
% in the network and the corresponding loss.

%% Specify Training Options
% Train for some epochs with a mini-batch size.

nbItperEpoch = ceil(nDij / miniBatchSize)
nbItMax = min(iterMaxUser, nbItperEpoch*numEpochs)

useOnecycle = false;
applyDecay = false;
decay = 1e-3; % if applyDecay and numEpochBeforeAltern<inf, then apply dynamic decay
% if applyDecay and numEpochBeforeAltern=inf, then apply static decay
% else no decay
numEpochBeforeAltern = 0; % a partir de là on memorise les modeles et leurs sorties
pause(1)

%%
% Specify the options for optimization.

optimizer = 'ADAM'; % 'ADAM' 'RMSPROP' or 'SGDM' % Can change the optimiser here
velocity = [];
averageGrad = [];
averageSqGrad = [];

switch optimizer

    case 'SGDM'
        % SGDM optimiser params
        momentum = 0.9;
        initialLearnRate = 0.01;

    case 'ADAM'
        % ADAM optimiser params
        gradDecay = 0.9;%0.75;
        sqGradDecay = 0.99;%0.90;
        initialLearnRate = 0.01;

    case 'RMSPROP'
        initialLearnRate = 0.01;
        sqGradDecay = 0.90;

    otherwise
        error('??')
end

% nb epochs * minibatchsize # B
% pendant x iterations ADAM LR faible puis alternance LR en partant de alpha0*LR
dynamicLearningRate.alpha0 = @(decay,initialLearnRate,iter)(initialLearnRate/(1 + decay*iter)); %@(x)(1); %
dynamicLearningRate.B = numEpochs; % nepochs max
dynamicLearningRate.trainingBudget = 300; % 600; % ie B/M the longer the more the loss "converges"
dynamicLearningRate.M = dynamicLearningRate.B / dynamicLearningRate.trainingBudget;

%iteration = 1:1000;
%y = dynamicLearningRate.alpha0/2 * (cos (pi * mod(iteration-1,ceil(dynamicLearningRate.B/dynamicLearningRate.M))/ceil(dynamicLearningRate.B/dynamicLearningRate.M)) +1);
%figure,plot(y)

if useOnecycle
    % prepare learning rate
    % max iterations
    pcrt = 0.3;
    plateau = 0; % reste a 0.1
    fin = 0.2; % plat sur la fin
    LRmax = initialLearnRate;
    LRmin = LRmax/25;
    n1 = floor(2*nbItMax*pcrt);
    h1 = hanning(n1);
    h1 = h1(1:round(length(h1)/2)); % first phase
    n12 = floor(nbItMax*plateau);% milieu
    h12 = ones(n12,1);
    n3 = floor(nbItMax*fin);
    n2 = nbItMax-floor(n1/2)-n12-n3;
    h2 = hanning(2*n2);
    h2 = h2(floor(length(h2)/2+1):end); % second phase
    h3 = h2(end)*ones(floor(fin*nbItMax),1);
    learningRateSchedule = [h1;h12;h2;h3];
    % au lieu d'aller de 0 a 1 on va de LRMIN à LRMAX

    learningRateSchedule=learningRateSchedule*(LRmax-LRmin)+LRmin;
    figure,plot(learningRateSchedule),ylabel('One cycle LR')
    clear h1 h2 h3 n2 n3 n12 h12 h1 n1
end

% decay = 0.0001;%0.01;
executionEnvironment = "auto";

%accfun = dlaccelerate(@modelGradients); clearCache(accfun)
accfun = @modelGradients;

%% Train Model

% * Use the custom mini-batch preprocessing function |preprocessMiniBatch| (defined
% at the end of this example) to convert the labels to one-hot encoded variables.
% * Format the image data with the dimension labels |'SSCB'| (spatial, spatial,
% channel, batch). By default, the |minibatchqueue| object converts the data to
% |dlarray| objects with underlying type |single|. Do not add a format to the
% class labels.
% * Train on a GPU if one is available. By default, the |minibatchqueue| object
% converts each output to a |gpuArray| if a GPU is available.  Using a GPU requires
% Parallel Computing Toolbox™ and a supported GPU device. For information on supported
% devices, see <docid:distcomp_ug#mw_57e04559-0b60-42d5-ad55-e77ec5f5865f GPU
% Support by Release>.

% On utilise ce MB pour trouver les indices des images. dijds contient le
% dij et les indices i et j.
% Si on a une grosse base de données on a un IMDS. Comment accéder aux
% images dans IMDS ?
mbq = minibatchqueue(Dij_ds,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFormat','BC','OutputEnvironment',executionEnvironment);
% clear Dij_ds

%%
% Initialize the training progress plot.

figure
lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
ylim([0 inf])
set(lineLossTrain.Parent,'Yscale','log')
xlabel("Iteration"), ylabel("Loss")
yyaxis right, lineLR = animatedline('Color',[0.1 0.325 0.798]);
ylabel("Learning rate")
grid on

% figure
% c=jet; c=c(1:round(size(c,1)/4):end,:);
% ylim([0 inf])
% xlabel("Iteration")
% ylabel("CVI")
% line_DB = animatedline('Color',c(2,:));
% line_SIL = animatedline('Color',c(3,:));
% yyaxis right, line_CH = animatedline('Color',c(4,:));
% grid on
% legend('DB','SIL','CH','location','eastoutside');

figure
c=jet; c=c(1:round(size(c,1)/7):end,:);
lineTrain = animatedline('Color','g');
ylim([0 1])
xlabel("Iteration")
ylabel("ARI")
yyaxis right, lineEval = animatedline('Color','r');
ylim([0 1])
legend('Train','Test','location','eastoutside');

%%
% * Evaluate the model gradients, state, and loss using the |dlfeval| and |modelGradients|
% functions and update the network state.
% * Determine the learning rate for the time-based decay learning rate schedule.
% * Update the network parameters.
% * Display the training progress.

% Loop over epochs.
iteration = 0;
start = tic;
lossEtParam = zeros(numEpochs,3);
minloss = inf;
cfact = ceil(dynamicLearningRate.B/dynamicLearningRate.M);
prevLearnRate = inf;
SILmax = -inf;
CHmax = -inf;
DBmin = +inf;

epoch = 1;
nbmodelesstored = 1;
continueEpochs = true;
kmodeles = 1;

iterlearn = 0;
learnRate = initialLearnRate;

nbFailed = 0;

do_evaluate_cvi = false;

if useOnecycle
    iterlearn = 1;
    learnRate = learningRateSchedule(1);
end

stocksLabelsBatches = [];

while continueEpochs

    reset(mbq);
    shuffle(mbq);

    % Loop over mini-batches.
    while hasdata(mbq) && iteration <= iterMaxUser
        iteration = iteration + 1;
        m = next(mbq);

        % read mini batch
        [dlXi, dlXj, dlDij, labelsXi, labelsXj, dlYi, dlYj, dl_idx_lig, dl_idx_col] = ...
            readMBQ(m, images_ds, minibasetraining, optionsEVNN);

        % assert(~any(isnan(dlXi),'all'));        assert(~any(isnan(dlXj),'all'))
        % assert(~any(isnan(dlDij),'all'));       assert(~any(isnan(labelsXi),'all'))
        % assert(~any(isnan(labelsXj),'all'));    assert(~any(isnan(dlYi),'all'))
        % assert(~any(isnan(dlYj),'all'));

        if any(isnan(dlXi), 'all') || any(isnan(dlXj), 'all') || ...
                any(isnan(dlDij), 'all') || any(isnan(labelsXi), 'all') || ...
                any(isnan(labelsXj), 'all') || any(isnan(dlYi), 'all') || ...
                any(isnan(dlYj), 'all')
            break;
        end

        % on stocke les labels pour voir au cas ou si tous les labels ont
        % ete presentés à l'algorithme
        if nbItMax < nbItperEpoch
            stocksLabelsBatches = [stocksLabelsBatches ; [dlYi,dlYj]];
        end

        % recherche des cas ou on a un a priori
        if ~isempty(priorIdx)
            fxi = ismember(gather(extractdata(dl_idx_lig)), priorIdx);
            fxi = gpuArray(dlarray(fxi,'CB'));
            fxj = ismember(gather(extractdata(dl_idx_col)), priorIdx);
            fxj = gpuArray(dlarray(fxj,'CB'));
        else
            fxi = []; 
            fxj = [];
        end

        % create intermediate matrix for current minibatch
        npairesMB = length(dlDij);
        mtC = zeros(optionsEVNN.fs,npairesMB);
        mtC = dlarray(double(mtC),'CB');
        if canUseGPU, mtC = gpuArray(mtC); end

        % create also the project matrix for conflict calculation
        Cmatrix = buildCmatrix22(optionsEVNN);
        Cmatrix = dlarray(Cmatrix);

        if 0
            % on s'attend a ce que les Dij pour des images de même classe
            % soient grands ?
            % cherche les labels où les Xi sont de classe c ET Xj de classe c aussi
            f11=find((dlYi==1 & dlYj==1) | (dlYi==2 & dlYj==2) | (dlYi==3 & dlYj==3));
            figure, histogram(extractdata(dlDij(f11))),s{1}=('Dij pour images dans meme classe');

            % cherche les labels où les Xi sont de classes differentes
            f11=find((dlYi==1 & dlYj==2) | (dlYi==1 & dlYj==3) | (dlYi==2 & dlYj==3) | (dlYi==2 & dlYj==1) | (dlYi==3 & dlYj==1) | (dlYi==3 & dlYj==2));
            hold on, histogram(extractdata(dlDij(f11))),s{2}=('Dij pour images dans classes differentes');
            legend(s)
        end

        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function and update the network state.
        if optionsEVNN.useApriori
            LXi = labelsXi; LXj = labelsXj;
        else LXi = []; LXj = [];
        end

        [gradients,loss,state] = dlfeval(accfun,dlnet,...
            dlXi,dlXj,dlDij,mtC,optionsEVNN,LXi,LXj,Cmatrix,fxi,fxj);

        % Si batch norm !!
        if bnstate
            dlnet.State = state;
        end

        % Determine learning rate for time-based decay learning rate schedule.
        %learnRate = initialLearnRate/(1 + decay*iteration);
        if useOnecycle
            learnRate=learningRateSchedule(iterlearn);
            iterlearn = iterlearn+1;
        else
            if applyDecay
                iterlearn = iterlearn+1;
                if epoch >= numEpochBeforeAltern % DYNAMIC
                    %learnRate = initialLearnRate * dynamicLearningRate.alpha0/2 * (cos (pi * mod(iterlearn-1,cfact)/cfact) +1);
                    a0 = dynamicLearningRate.alpha0(decay,initialLearnRate,iterlearn);
                    learnRate = a0/2 * (cos (pi * mod(iterlearn-1,cfact)/cfact) +1);
                else
                    learnRate = initialLearnRate/(1 + decay*iterlearn);
                end
            end
        end

        loss = double(extractdata(loss));

        % update model
        switch optimizer

            case 'ADAM' % Update the network parameters using the adamupdate function.
                [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,gradients,averageGrad, ...
                    averageSqGrad,iteration,learnRate,gradDecay,sqGradDecay);

            case 'RMSPROP'
                [dlnet,averageSqGrad] = rmspropupdate(dlnet,gradients,averageSqGrad,learnRate, sqGradDecay);

            case 'SGDM'
                [dlnet,velocity] = sgdmupdate(dlnet,gradients,velocity,learnRate,momentum);
        end

        % ###########################################
        % Store the model if it improves the criterion (unsupervised)
        % Evaluate first the model on data (does not use Ytrain)
        [ARIplTrain,ARIplEval,ACC,NMI,ARIbetp,ARIbba,RIpl,RIbetp,RIbba,~,Ypredpl,pl,~,betp,outputNN] = ...
            testModel(dlXi,dlXj,dlYi,dlYj,dlnet,optionsEVNN,false,eval,test);
        if any(isnan(outputNN),'all')
            break;
        end


        if do_evaluate_cvi

            [DBcourant, CHcourant, SILcourant, ...
                ARIpl, ARIbetp, ARIbba, RIpl, RIbetp, RIbba, NMI,...
                cmpl, cmbetp, cmbba] = ...
                evaluate_cvi(dlnet, trainingFeaturesHOG_PCA, minibasetraining, ...
                optionsEVNN, false,eval);

            addpoints(line_DB,iteration,DBcourant)
            addpoints(line_SIL,iteration,SILcourant)
            addpoints(line_CH,iteration,CHcourant)
            if DBcourant <= DBmin
                DBmin_ari = [ARIpl,ARIbetp,ARIbba,iteration,epoch];
                DB_cmpl = cmpl;
                DB_cmbetp = cmbetp;
                DB_cmbba = cmbba;
            end
            if CHcourant >= CHmax
                CHmax_ari = [ARIpl,ARIbetp,ARIbba,iteration,epoch];
                CH_cmpl = cmpl;
                CH_cmbetp = cmbetp;
                CH_cmbba = cmbba;
            end
            if SILcourant >= SILmax
                SILmax_ari = [ARIpl,ARIbetp,ARIbba,iteration,epoch];
                SIL_cmpl = cmpl;
                SIL_cmbetp = cmbetp;
                SIL_cmbba = cmbba;
            end

        end

        % Display the training progress.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain,iteration,loss)
        addpoints(lineLR,iteration,learnRate)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow

        if loss<minloss, minloss = loss; save dlnetEVNN dlnet iteration loss optimizer averageSqGrad averageGrad velocity learnRate; end
        addpoints(lineTrain,iteration,ARIplTrain)
        addpoints(lineEval,iteration,ARIplEval)

        lossEtParam (iteration,:) = [loss,ARIplTrain,ARIbetp];
        nbFailed = 0;

    end

    epoch=epoch+1;
    if epoch > numEpochs  || iteration >= iterMaxUser
        if iteration >= iterMaxUser
            disp('Iteration max reached')
        end
        continueEpochs = false;
    end
end

%% Evaluate
disp('Fin')
dlXi = test.images;
dlXi = dlarray(single(dlXi),'SSBC');
dlYi = test.labels;
dlFTestXi = predict(dlnet,dlXi);
applyFWd = true;
[masses,~,betp,pl] = weights2MassesSoftmax22(dlFTestXi,optionsEVNN,applyFWd);
pl=gather(extractdata(pl));
[~,Ypredpl] = max(double(pl));
[ARIpltest,~,~,~,Cpltest] = valid_RandIndex(Ypredpl,dlYi);
NMItest = nmi(gather(Ypredpl),dlYi);
ACCtest = cluster_acc(dlYi,gather(Ypredpl));
disp(ACCtest)
disp(NMItest)
disp(ARIpltest)


%% Test Model
% Test the classification accuracy of the model by comparing the predictions
% on the validation set with the true labels.

% [DBcourant, CHcourant, SILcourant, ...
%     ARIpl, ARIbetp, ARIbba, RIpl, RIbetp, RIbba, NMI,...
%     cmpl, cmbetp, cmbba] = ...
%     evaluate_cvi(dlnet, trainingFeaturesHOG_PCA, minibasetraining, optionsEVNN, true,eval);

% % charge le reseau pour la loss mini
% %ll=load('dlnetEVNN');
% %fprintf('Loss min %f pour iter %d\n',ll.loss,ll.iteration)
% % charge les données images et les passer dans le reseau
% [a,b] = unique(lig);
% dlXi = readByIndex(images_ds,a);
% dlYi = cell2mat(dlXi.response);
% dlXi = dlXi.input;
% dlXi = cell2mat(dlXi); % figure,image(dlXi(1:28,1:28))
% dlXi = reshape(dlXi,28,[],28);
% %figure,image(squeeze(dlXi(:,ii,:)))
% dlXi = permute(dlXi,[1 3 2]);
% %figure,image(squeeze(dlXi(:,:,ii)))
% dlXi = dlarray(single(dlXi),'SSBC');
% % pass features of the test data in the net
% dlFTestXi = forward(dlnet,dlXi); % predict ???
%
% % get bba
% applyFWd = true;
% [masses,~,betp,pl] = weights2MassesSoftmax22(dlFTestXi,optionsEVNN,applyFWd);
%
% % predict using BBA on singletons
% m = double(extractdata(masses));  % BBA
% m = m(2:optionsEVNN.numClasses+1,:);
% [~,Ypredbba] = max(m);
% % using pl
% [~,Ypredpl] = max(double(extractdata(pl)));
% % using betp
% [~,Ypredbetp] = max(double(extractdata(betp)));
%
% % ARI perf
% [ARIpl,RIpl,~,~,Cpl] = valid_RandIndex(Ypredpl,dlYi);
% [ARIbetp,RIbetp,~,~,Cbetp] = valid_RandIndex(Ypredbetp,dlYi);
% [ARIbba,RIbba,~,~,Cbba] = valid_RandIndex(Ypredbba,dlYi);
%
% % for confusion matrix
% z = zeros(optionsEVNN.numClasses);
% z(1:size(Cpl,1),1:size(Cpl,2)) = Cpl;
%
% figure, confusionchart(z'); title('PL confusion matrix');
% figure,subplot(4,2,1)
% plot(extractdata(masses(1,:))),yyaxis right,hold on,plot(dlYi),title('Conflit')
% for i=2:4
%     subplot(4,2,i)
%     plot(extractdata(masses(i,:))),yyaxis right,hold on,plot(dlYi),title(['m(' num2str(i) ')'])
% end
% for i=5:7
%     subplot(4,2,i)
%     plot(extractdata(masses(i,:))),yyaxis right,hold on,plot(dlYi),title(['couple' num2str(i)])
% end
% subplot(4,2,8)
% plot(extractdata(masses(8,:))),yyaxis right,hold on,plot(dlYi),title('Omega')


%% Model Gradients Function
% The |modelGradients| function takes a |dlnetwork| object |dlnet|, a mini-batch
% of input data |dlX| with corresponding labels |Y| and returns the gradients
% of the loss with respect to the learnable parameters in |dlnet|, the network
% state, and the loss. To compute the gradients automatically, use the |dlgradient|
% function.

function [gradients,loss,state] = modelGradients(net,...
    dlXi,dlXj,dlDij,mtC,optionsEVNN,labelsXi,labelsXj,Cmatrix, idxFXi, idxFXj)

% ON FAIT PASSER LES DATA DANS LE RESEAU ET ON CALCULE LA LOSS
n = size(dlXi,4);

% PROBLEME D'UNICITE DES INSTANCES en effet le batchnorm va voir passer
% plusieurs la meme instance ici donc on va biaiser le mean et std ??
[Fijnew,state] = forward(net,cat(4,dlXi,dlXj));

% Extract masses from output, compute C matrix for conflict calculation
% pl useful if labels provided
if (~isempty(labelsXi) && sum(idxFXi)>0) || (~isempty(labelsXj) && sum(idxFXj)>0)

    [Fijnew,~,~,predictXipl] = weights2MassesSoftmax22(Fijnew,optionsEVNN,true);
    predictXjpl = predictXipl(:,n+1:end);
    predictXipl = predictXipl(:,1:n);

else

    Fijnew = softmax(Fijnew);
    predictXipl = [];
    predictXjpl = [];

end

% Compute Kij - version rapide peu de boucles for mais memory gourmande
for i=1:size(Cmatrix,2)
    mtC(i,:) = sum(Fijnew(:,1:n) .* Cmatrix(:,i), 1);
end
Kij = sum(mtC .* Fijnew(:,n+1:end),1);


% Compute loss
loss = computeLoss_bba2(Kij, dlDij,...
    predictXipl, predictXjpl, labelsXi, labelsXj,...
    idxFXi, idxFXj, optionsEVNN);

% Gradients
gradients = dlgradient(loss,net.Learnables);

end

%% Read MBQ

function [dlXi, dlXj, dlDij, labelsXi, labelsXj, dlYi, dlYj, dl_idx_lig, dl_idx_col] = ...
    readMBQ(m, images_ds, minibasetraining, optionsEVNN)

% m contains Dij, i and j
dlDij = m(1,:);
dl_idx_lig = m(2,:);
dl_idx_col = m(3,:);

% get images
% if 0
%
%     %ii=109
%     dlXi = readByIndex(images_ds,dl_idx_lig);
%     dlYi = cell2mat(dlXi.response);
%     %figure,image(dlXi.input{ii})
%     dlXi = dlXi.input;
%     dlXi = cell2mat(dlXi); % figure,image(dlXi(1:28,1:28))
%     dlXi = reshape(dlXi,28,[],28);
%     %figure,image(squeeze(dlXi(:,ii,:)))
%     dlXi = permute(dlXi,[1 3 2]);
%     %figure,image(squeeze(dlXi(:,:,ii)))
%
%     dlXj = readByIndex(images_ds,dl_idx_col);
%     dlYj = cell2mat(dlXj.response);
%     dlXj = dlXj.input;
%     dlXj = cell2mat(dlXj);
%     dlXj = reshape(dlXj,28,[],28);
%     dlXj = permute(dlXj,[1 3 2]);
%
% else

% dl_idx contiennent les indices des images (lig et col pour les paires)
dlXi = minibasetraining.images(:,:,dl_idx_lig);
dlYi = minibasetraining.labels(dl_idx_lig);
dlXj = minibasetraining.images(:,:,dl_idx_col);
dlYj = minibasetraining.labels(dl_idx_col);

% end

if ~isempty(optionsEVNN.imresize)
    dlXiresized = zeros([optionsEVNN.imresize size(dlXi,3)]);
    for i=1:size(dlXi,3)
        dlXiresized(:,:,i) = imresize(dlXi(:,:,i),optionsEVNN.imresize);
    end
    dlXi = dlXiresized; clear dlXiresized

    dlXjresized = zeros([optionsEVNN.imresize size(dlXj,3)]);
    for i=1:size(dlXj,3)
        dlXjresized(:,:,i) = imresize(dlXj(:,:,i),optionsEVNN.imresize);
    end
    dlXj = dlXjresized; clear dlXjresized
end

dlXi = dlarray(single(dlXi),'SSBC');
dlXj = dlarray(single(dlXj),'SSBC');
dlDij = dlarray(dlDij,'CB');

labelsXi = dlarray(labels2matrix(dlYi,optionsEVNN.numClasses),'BC');
labelsXj = dlarray(labels2matrix(dlYj,optionsEVNN.numClasses),'BC');

end

%% Evaluate
% Use the current model and evaluate the quality using CVI.

function [DBcourant, CHcourant, SILcourant, ...
    ARIpl, ARIbetp, ARIbba, RIpl, RIbetp, RIbba, NMI, cmpl, cmbetp, cmbba] = ...
    evaluate_cvi(dlnet, trainingFeatures, minibasetraining, optionsEVNN, tracerRes,eval)

dlFeats = trainingFeatures;

if 0
    dlXi = readByIndex(images_ds,uniqueLig);
    dlYi = cell2mat(dlXi.response);
    dlXi = dlXi.input;
    dlXi = cell2mat(dlXi); % figure,image(dlXi(1:28,1:28))
    dlXi = reshape(dlXi,28,[],28);
    %figure,image(squeeze(dlXi(:,ii,:)))
    dlXi = permute(dlXi,[1 3 2]);
    %figure,image(squeeze(dlXi(:,:,ii)))
    dlXi = dlarray(single(dlXi),'SSBC');

else
    dlXi = minibasetraining.images;
    dlXi = dlarray(single(dlXi),'SSBC');
    dlYi = minibasetraining.labels;

end

% pass features of the test data in the net
dlFTestXi = predict(dlnet,dlXi); % predict ???

if 1
    % GRADCAM
    % Voir https://fr.mathworks.com/help/deeplearning/ref/gradcam.html
    % The reductionFcn function receives the output from the reduction layer as a
    % traced dlarray object. The function must reduce this output to a scalar dlarray,
    % which gradCAM then differentiates with respect to the activations of the feature layer.
    % For example, to compute the Grad-CAM map for channel 208 of the softmax activations
    % of a network, the reduction function is @(x)(x(208)). This function receives the
    % activations and extracts the 208th channel.

    n=5;
    disp('Label de la base')
    disp(minibasetraining.labels(n))
    disp('Label vrai visible sur l''image')
    disp(minibasetraining.labels(n)-1)
    disp('Label attendu sortie de reseau')
    labelattendu = floor(n/100)+1;%9 % depend en fait du numero du cluster, tracer la matrice de confusion avant
    labelattendu = 1;
    disp(labelattendu)
    % figure,imshow(minibasetraining.images(:,:,n))
    ss=[];
    % depend en fait du numero du cluster, tracer la matrice de confusion avant
    for labelattendu=labelattendu%1:10
        disp(labelattendu)
        % figure,imshow(minibasetraining.images(:,:,n))
        featureLayer = 'conv_2'; % couche d'activation
        reductionLayer = 'fc';% couche de reduction, sortie ?
        reductionFcn = @(x)(x(labelattendu)); % prend la sortie correspondant à la
        % classe predite / cluster, attention + 1 du fait du conflit, SI MASS!!
        % reductionFcn = @(x)(mtopl_mnistdigits(x,labelattendu,optionsEVNN));
        % voir si cela a du sens de regarder les classes de doute ?
        % il faudrait sortir la pl ! mais il faut une fonction
        % differentiable...
        scoreMap = gradCAM(dlnet,dlXi(:,:,1,n),reductionFcn, ...
            'ReductionLayer',reductionLayer, ...
            'FeatureLayer',featureLayer);

        % relu - negatif reponse => bad
        scoreMap(scoreMap<0)=0;

        ss=[ss    sum(scoreMap,'all')]

        figure, subplot(121)
        imshow(imresize(rescale(extractdata(dlXi(:,:,1,n))),[224 224]))
        %hold on, imagesc(imresize(rescale(scoreMap),[224 224]),'AlphaData',0.5)
        subplot(122), imagesc(imresize(rescale(scoreMap),[224 224]))
        % colormap jet
    end

    % a faire
    % We can average together Grad-CAM heat-maps from all model layers.
end

% get bba
applyFWd = true;
[masses,~,betp,pl] = weights2MassesSoftmax22(dlFTestXi,optionsEVNN,applyFWd);
pl=gather(extractdata(pl));
masses=gather(extractdata(masses));
betp=gather(extractdata(betp));

% predict using BBA on singletons
m = double(masses);  % BBA
m = m(2:optionsEVNN.numClasses+1,:);
[~,Ypredbba] = max(m);
% using pl
[~,Ypredpl] = max(double(pl));
% using betp
[~,Ypredbetp] = max(double(betp));

% perf values
NMI = nmi(gather(Ypredpl),dlYi);
[ARIbetp,RIbetp,~,~,Cbetp] = valid_RandIndex(Ypredbetp,dlYi);
[ARIbba,RIbba,~,~,Cbba] = valid_RandIndex(Ypredbba,dlYi);

% Evaluate quality, using DB for ex, on pl for example
DBcourant = evalclusters(double(gather(dlFeats)),double(gather(Ypredpl))','DaviesBouldin'); % MIN better
DBcourant = DBcourant.CriterionValues;
CHcourant = evalclusters(double(gather(dlFeats)),double(gather(Ypredpl))','CalinskiHarabasz'); % MAX better
CHcourant = CHcourant.CriterionValues;
SILcourant = evalclusters(double(gather(dlFeats)),double(gather(Ypredpl))','silhouette'); % MAX better
SILcourant = SILcourant.CriterionValues;

% for confusion matrix
z = zeros(optionsEVNN.numClasses);
z(1:size(Cpl,1),1:size(Cpl,2)) = Cpl;

figure, confusionchart(z'); title(sprintf('PL confusion matrix as provided ARI = %f',ARIpl));

cmpl = z;
cmbetp = zeros(optionsEVNN.numClasses);
cmbetp(1:size(Cbetp,1),1:size(Cbetp,2)) = Cbetp;
cmbba = zeros(optionsEVNN.numClasses);
cmbba(1:size(Cbba,1),1:size(Cbba,2)) = Cbba;

if tracerRes

    % inferred_labels = infer_cluster_labels(Ypredpl, dlYi, optionsEVNN.numClasses);
    % predicted_labels = infer_data_labels(Ypredpl, inferred_labels);
    % predicted_labels = predicted_labels(:);
    %
    % [ARIpl,RIpl,~,~,Cpl] = valid_RandIndex(predicted_labels,dlYi); % the same as ari(Ypredpl,dlYi) with correct order
    % figure,confusionchart(confusionmat(dlYi(:),predicted_labels)),title('PL confusion AFTER reassignment');
    %
    % % liste des images mal classees
    % for i=1:optionsEVNN.numClasses
    %     % truth is i and predicted is not i
    %     f = (dlYi == i) & (predicted_labels ~= i);
    %     g=find(f);
    %     s=[];
    %     if ~isempty(g)
    %         figure,imshow(imtile(minibasetraining.images(:,:,f)))
    %         for j=1:length(g), s=[s, '-' num2str(g(j))]; end, s(1)='';
    %         %else no mistakes !
    %         title(sprintf('Predicted sth else, while truth is class %d \n Wrongly classified %s',i,s))
    %         drawnow
    %     end
    % end

    s = round(size(masses,1)/2);
    r = mod(size(masses,1),2);
    nl = s+r;
    figure,subplot(nl,2,1)
    plot((masses(1,:))),yyaxis right,hold on,plot(dlYi),title('Conflit')
    k=1;
    for i=2:optionsEVNN.numClasses+1
        subplot(nl,2,i)
        plot((masses(i,:))),yyaxis right,hold on,plot(dlYi),title(['m(' num2str(k) ')'])
        k=k+1;
    end
    k=optionsEVNN.numClasses+2;
    for i=1:optionsEVNN.numClasses
        for j=i+1:optionsEVNN.numClasses
            subplot(nl,2,k)
            plot((masses(k,:))),yyaxis right,hold on,plot(dlYi),
            title(['couple' num2str(i) ' U ' num2str(j)])
            k=k+1;
        end
    end
    subplot(nl,2,size(masses,1))
    plot((masses(end,:))),yyaxis right,hold on,plot(dlYi),title('Omega')

    s = round(size(pl,1)/2);
    r = mod(size(pl,1),2);
    nl = s+r;
    figure,subplot(nl,2,1)
    k=1;
    for i=1:optionsEVNN.numClasses
        subplot(nl,2,i)
        plot((pl(i,:))),yyaxis right,hold on,plot(dlYi),title(['pl(' num2str(k) ')'])
        k=k+1;
    end

end

end

%% Test

function [ARIplTrain, ARIplEval,ACC,NMI,ARIbetp,ARIbba,RIpl,RIbetp,RIbba,z,Ypredpl,pl,masses,betp,outputNN] = ...
    testModel(dlXi,dlXj,dlYi,dlYj,dlnet,optionsEVNN,plotfigure,eval,test)
% Returns ARI based on pl and betp, z is the confusion matrix and Ypredpl
% the labels predicted based on pl, returns pl on singletons and full bba
% dlFTest is the output of the NN without softmax

% pass features of the test data in the net
dlFTestXi = predict(dlnet,dlXi); % predict ???
dlFTestXj = predict(dlnet,dlXj); % predict ???
outputNN = [dlFTestXi,dlFTestXj];

% get bba
applyFWd = true;
[massesXi,~,betpXi,plXi] = weights2MassesSoftmax22(dlFTestXi,optionsEVNN,applyFWd);
[massesXj,~,betpXj,plXj] = weights2MassesSoftmax22(dlFTestXj,optionsEVNN,applyFWd);

pl=[plXi,plXj];
masses=[massesXi,massesXj];
betp=[betpXi,betpXj];

Ytest=[dlYi;dlYj]';
% Ytest=Ytest+1; % avoid 0 index

% predict using BBA on singletons
m = double(extractdata(masses));  % BBA
m = m(2:optionsEVNN.numClasses+1,:);
[~,Ypredbba] = max(m);
% using pl
[~,Ypredpl] = max(double(extractdata(pl)));
% using betp
[~,Ypredbetp] = max(double(extractdata(betp)));

% perf values
% NMI = nmi(gather(Ypredpl),Ytest);
[ARIplTrain,RIpl,~,~,Cpl] = valid_RandIndex(Ypredpl,Ytest);
[ARIbetp,RIbetp,~,~,Cbetp] = valid_RandIndex(Ypredbetp,Ytest);
[ARIbba,RIbba,~,~,Cbba] = valid_RandIndex(Ypredbba,Ytest);
% NMI = nmi(gather(Ypredpl),Ytest);
% ACC=cluster_acc(Ytest,gather(Ypredpl));
minibasetraining.images=test.images;
minibasetraining.labels=test.labels+1;
dlXi = minibasetraining.images;
dlXi = dlarray(single(dlXi),'SSBC');
dlYi = minibasetraining.labels;
dlFTestXi = predict(dlnet,dlXi);
applyFWd = true;
[masses,~,betp,pl] = weights2MassesSoftmax22(dlFTestXi,optionsEVNN,applyFWd);
pl=gather(extractdata(pl));
[~,Ypredpl] = max(double(pl));
[ARIplEval,RIpl,~,~,Cpl] = valid_RandIndex(Ypredpl,dlYi);
NMI = nmi(gather(Ypredpl),dlYi);
ACC=cluster_acc(dlYi,gather(Ypredpl));

% for confusion matrix
z = zeros(optionsEVNN.numClasses);
z(1:size(Cpl,1),1:size(Cpl,2)) = Cpl;
% disp(z)

% fprintf('ARIpl=%2.2f\n', ARIpl);

if plotfigure
    figure, confusionchart(z'); title('PL confusion matrix');
end

end

%% And another one

function loss = computeLoss_bba2(Kij, Dij, predictXi, predictXj, ...
    labelsXi, labelsXj, idxFXi, idxFXj, optionsEVNN)
% Calculates the loss between the Dij and conflict
% See https://arxiv.org/pdf/1701.03077.pdf

switch optionsEVNN.loss_type.type
    case {'mse' 'l2loss'}
        loss = mean((Kij - Dij).^2);
        %loss = l2loss(Kij,Dij);
        % lossMSE = 2*mse(Kij,Dij);
        %case 'crossent' % no sense for regression
        %https://fr.mathworks.com/matlabcentral/answers/151699-am-i-computing-cross-entropy-incorrectly
        %    loss = -mean( Dij .* log( Kij + double(Kij==0)) );
    case 'l1loss'
        %loss = l1loss(Kij,Dij);
        loss = mean(abs(Kij-Dij));
    case 'huber'
        %loss = huberLoss(Kij, Dij, optionsEVNN.loss_type.param);
        loss = huber(Kij, Dij, 'TransitionPoint',optionsEVNN.loss_type.param);
    case 'pseudohuber'
        loss = pseudohuberLoss(Kij, Dij, optionsEVNN.loss_type.param);
    case 'smoothL1Loss'
        loss = smoothL1Loss(Kij, Dij, optionsEVNN.loss_type.param);
    case 'e_mae'
        loss = epsil_insensitive_mae(Kij, Dij, optionsEVNN.loss_type.param);
    case 'logcosh'
        loss = logcosh(Kij, Dij);
    case 'qreg'
        loss = quantileregressionloss(Kij, Dij, optionsEVNN.loss_type.param);
    case 'oneMinusExp'
        loss = mean(1 - exp(-optionsEVNN.loss_type.param*abs(Kij-Dij)));
end

% case supervised
lossCEnt = 0.0;

if ~isempty(labelsXi) && sum(idxFXi)>0
    lossCEnt = mean( ( predictXi(:,idxFXi) - labelsXi(:,idxFXi) ).^2, 'all');
    %lossCEnt = mean( sum ( ( predictXi(:,idxFXi) - labelsXi(:,idxFXi) ).^2, 1 ), 2);

    %P = predictXi(:,idxFXi) .* labelsXi(:,idxFXi);
    %lossCEnt = -mean( log(P + double(P==0)), 'all');

    %P = predictXi(:,idxFXi);
    %P = log(P + double(P==0));
    %P = P .* labelsXi(:,idxFXi);
    %lossCEnt = -sum(P, 'all')/sum(idxFXi);

end
if ~isempty(labelsXj)  && sum(idxFXj)>0
    lossCEnt = lossCEnt + mean( ( predictXj(:,idxFXj) - labelsXj(:,idxFXj) ).^2, 'all');
    %lossCEnt = lossCEnt + mean( sum ( ( predictXj(:,idxFXj) - labelsXj(:,idxFXj) ).^2, 1 ), 2);

    %P = predictXj(:,idxFXj) .* labelsXj(:,idxFXj);
    %lossCEnt = lossCEnt - mean( log(P + double(P==0)), 'all');

    %P = predictXj(:,idxFXj);
    %P = log(P + double(P==0));
    %P = P .* labelsXj(:,idxFXj);
    %lossCEnt = lossCEnt - sum(P, 'all')/sum(idxFXj);

end

p1 = 1;
p2 = 0;
loss = p1*loss/(p1+p2) + p2*lossCEnt/(p1+p2);

% AJOUT NON SPECIFICITE
% numclasses = optionsEVNN.numClasses;
% usedoute = optionsEVNN.usedoute; assert(usedoute)
% usepaires = optionsEVNN.usepaires;
% positionDoute = numclasses + 2; % conflict first
% positionClasse1 = 2;
% positionClasseK = numclasses + 1;
%ui = unique(extractdata(dl_idx_lig));
%uj = unique(extractdata(dl_idx_col));
%uniqueData = union(ui,uj); % unique
%uniqueFijnew = Fijnew(:,uniqueData); % ?? fct de Fijnew, on n'a pas besoin de le faire pour tous les couples, seules pour les Xi, i=1:N
% m = Fijnew ./ (1-Fijnew(1,:));
% m(1,:) = 0;
% if usepaires
% 	L = mean( sum( m(positionDoute+1:end, :) * log2(2), 1 ) ...
%         + m(positionDoute, :)*log2(numclasses), 2);
% else
% 	L = mean( m(positionDoute, :)*log2(numclasses), 2);
% end
% L = L / log2(numclasses);
%
% loss = loss + 0.1*L;

end


function h = huberLoss(Kij,Dij,th)

assert(th > 0);

h = abs(Kij-Dij);
f = h < th;
h(f) = 0.5*h(f).^2;
h(~f) = th*( h(~f) - 0.5*th);
h = mean(h,'all');

end

function h = pseudohuberLoss(Kij, Dij, th)

assert(th > 0);

h = mean(th^2 * (sqrt( 1 + ( (Kij-Dij) / th ).^2 ) - 1));

end


function h = smoothL1Loss(Kij, Dij, th)

assert(th > 0);

h = mean((sqrt( 1 + ( (Kij-Dij) / th ).^2 ) - 1));

end


function h = epsil_insensitive_mae(Kij,Dij,e)

h = mean(max(0,abs(Kij-Dij)-e));

end

function h = logcosh(Kij, Dij)

h = sum(log(cosh(Kij - Dij)));

end

function h = quantileregressionloss(Kij, Dij, g)

assert(g>=0); assert(g<=1);

h = Dij - Kij;
f = h < 0;
p1 = sum( (1-g)*abs(h(f)) );
p2 = sum( g*abs(h(~f)) );

h = p1 + p2;

end