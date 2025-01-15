


X1 = Connectome.(Weights(weight));
Y = TargetNumeric;
Index = ~isnan(Y);
Y = Y(Index);
if phenotype == 2
    Y = categorical(Y);
end
L = sum(Index);
TestI = 1:1230;
I = 1231:L;
if phenotype == 4
    TestI = 1:940;
    I = 941:L;
end
[partition] = Partition(length(I),5);
CV = ["CV1", "CV2", "CV3", "CV4", "CV5"];
X = reshape(X1(:,:,Index), [M M 1 L]);
clear X1
if phenotype > 2
    One = repmat(ones(M),1,1,L);
    Sex = GetCovar(Index,2);
    Sex(Sex == 0) = -1;
    Age = GetCovar(Index,1);
    normedAge = (Age - mean(Age(I)))./ std(Age(I));
    X(:,:,2,:)=One.*(reshape(Gender,[1 1 L]));
    X(:,:,3,:)=One.*(reshape(Age,[1 1 L]));
end
if phenotype == 2
    results = zeros(3,length(CV));
else
    results = zeros(6,length(CV));
end
GradientMap = zeros(M,M,length(CV));
for cv=1:5
    if ~exist(CV(cv),'dir')
        mkdir(CV(cv))
    end
    CVpathfiles = strcat(CV(cv),"\*.mat");
    checkout = strcat(path,"\",Weights(weight),"\",CV(cv));
    disp(strcat("Training BrainNetCNN: ",FolderName,"\",Weights(weight),"\",CV(cv)));
    XTest = X(:,:,:,TestI);
    YTest = Y(TestI);
    XValidation = X(:,:,:,I(partition{2,cv}));
    YValidation = Y(I(partition{2,cv}),:);
    XTrain = X(:,:,:,I(partition{1,cv}));
    YTrain = Y(I(partition{1,cv}));
    options1 = trainingOptions('adam','Shuffle','every-epoch', 'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',50,'MaxEpochs',200,'MiniBatchSize',128,'InitialLearnRate',0.001,'LearnRateSchedule','piecewise', ...
'LearnRateDropFactor',0.9,'LearnRateDropPeriod',20,'OutputNetwork','best-validation-loss',...
'ExecutionEnvironment','gpu','L2Regularization',1e-6,'CheckpointPath',checkout,'Verbose',false);
    if isempty(dir(CVpathfiles)) %Do not have the DL progress
        rng('default')
        [finalnetforgrad] = trainNetwork(XTrain,YTrain,lgraph,options1);
    end

    disp(strcat("Get Prediction Results: ",FolderName,"\",Weights(weight),"\",CV(cv)));
    Accuracy_Epoch_Plot_inCV

    clear XTrain XTest XValidation YTrain YTest YValidation 
    clear testAccuracy trainingAccuracy ValidationAccuracy
    clear testCorr trainingCorr ValidationCorr
    clear trainingloss files

    disp(strcat("Computing Gradients: ",FolderName,"\",Weights(weight),"\",CV(cv)));
    ComputingGradients_inCV

    GradientMap(:,:,cv) = MeanGrad;
    AllGradNet.(Phenotypes(phenotype)).(Weights(weight)).Net(i) = net;

    clear net
end

if size(results,1) == 3
    resultcolumn = [mean(results,2);std(results,[],2)];
else
    resultcolumn = mean(results,2);
end