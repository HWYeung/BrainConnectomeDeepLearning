
filepath = checkout;
filespath =  strcat(filepath,'\*.mat');
files=dir(filespath);
addpath(filepath);
filelength = length(files);
Date = string(zeros(200,1));
for epoch= 1:filelength 
    Date(epoch) = files(epoch).date;
end

[~,DateIndex] = sort(Date);
trainingAccuracy = zeros(filelength ,1);
ValidationAccuracy = zeros(filelength ,1);
testAccuracy = zeros(filelength ,1);
trainingloss = zeros(filelength,1);
if phenotype == 2
    Predfun = @(x,y) classify(x,y,'ExecutionEnvironment','gpu');
    ACCfun = @(x,y) mean(x==y)*100;
    Lossfun = @(x,y,z) mean((double(z) == [1 2]).*log(predict(x,y,'ExecutionEnvironment','gpu')),'all');
else
    Predfun = @(x,y) predict(x,y,'ExecutionEnvironment','gpu');
    ACCfun = @(x,y) mean(abs(x-y));
    Corrfun = @(x,y) corr(x,y);
    Lossfun = @(x,y,z) mean((Predfun(x,y)-z).^2);
    trainingCorr = zeros(filelength ,1);
    ValidationCorr = zeros(filelength ,1);
    testCorr = zeros(filelength ,1);
end
for epoch= 1:filelength 
    Thisfile=files(DateIndex(epoch)).name;
    load(Thisfile);
    options = trainingOptions('sgdm', 'InitialLearnRate',eps, 'ResetInputNormalization',false,'OutputFcn',@(~)true,'Verbose',false );
    net = trainNetwork(XTrain,YTrain,layerGraph(net) ,options);
    YPredTrain = Predfun(net,XTrain);
    trainingAccuracy(epoch) = ACCfun(YTrain,YPredTrain);
    trainingloss(epoch) = Lossfun(net,XTrain,YTrain);
    YPredValid = Predfun(net,XValidation);
    ValidationAccuracy(epoch) = ACCfun(YValidation,YPredValid);
    YPredTest = Predfun(net,XTest);
    testAccuracy(epoch) = ACCfun(YTest,YPredTest);
    if phenotype ~= 2
        trainingCorr(epoch) = corr(YTrain , YPredTrain);
        ValidationCorr(epoch) = corr(YValidation , YPredValid);
        testCorr(epoch) = corr(YTest , YPredTest);
    end
    clear net YPredValid YPredTest YPredTrain
end
% The rows of accuracies can be used to plot the training progress
if phenotype == 2
    Cost = ValidationAccuracy.*(trainingAccuracy > ValidationAccuracy);
    maxValidate = find(ValidationAccuracy == max(Cost));
else
    Cost = ValidationCorr.*(trainingCorr > ValidationCorr).*(trainingAccuracy < ValidationAccuracy);
    maxValidate = find(ValidationCorr == max(Cost));
end

%}
Thisfile=files(DateIndex(maxValidate(end))).name;
load(Thisfile);

results(1,cv) = trainingAccuracy( maxValidate(end));
results(2,cv) = ValidationAccuracy( maxValidate(end));
results(3,cv) = testAccuracy( maxValidate(end));
if phenotype ~=2
    results(4,cv) = trainingCorr( maxValidate(end));
    results(5,cv) = ValidationCorr( maxValidate(end));
    results(6,cv) = testCorr( maxValidate(end));
end


rmpath(filepath);