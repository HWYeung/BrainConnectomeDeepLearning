addpath(genpath('UKB_DL_ML_Comparison'));

load("Data.mat");

GetCovar = TargetVariables(:,1:2);
% TargetVariables columns go in the order: Age, Sex, GFactor, MHQFactor
% Take away training mean from the age variable
MeanAge = mean(TargetVariables(1231:end,1));
TargetVariables(:,1) = TargetVariables(:,1) - MeanAge;

% Perform the training
Phenotypes = ["Age","Sex","GFactorWithCovariates","MHQFactorWithCovariates"];
Weights = ["MD","FA","SC","OD","ISOVF","ICVF"];
cd UKB_DL_ML_Comparison
addpath("Code")
currentpath = pwd;
M = 85; %node number
for phenotype = 1:4
    FolderName = strcat(Phenotypes(phenotype),"DLProgress");
    if ~exist(FolderName,'dir')
        mkdir(FolderName)
    end
    cd(FolderName)
    TargetNumeric = TargetVariables(:,phenotype);
    if ~isempty(intersect(phenotype, [1 2]))
        GraphCNN
    else
        GraphCNNWithCovariates
    end
    path = pwd;
    ResultTable = table();
    for weight = 1:length(Weights)
        if ~exist(Weights(weight),'dir')
            mkdir(Weights(weight))
        end
        cd(Weights(weight))
        BrainNetCNNModelTraining
        clear X
        ResultTable.(Weights(weight)) = resultcolumn;
        AllGradNet.(Phenotypes(phenotype)).(Weights(weight)).Grad = GradientMap;
        AllGradNet.(Phenotypes(phenotype)).(Weights(weight)).Net = net;
        cd ..
    end
    AllResults.(strcat(Phenotypes(phenotype),"PredictionTable")) = ResultTable;
    cd ..
    save(fullfile(currentpath,'DLResultsForAllPredictionTasks.mat'),'AllResults');
    save(fullfile(currentpath,'DLGradAndNetForAllPredictionTasks.mat'),'AllGradNet');
end


