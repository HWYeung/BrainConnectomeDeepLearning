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

%SolverOptions
%Ridge -> LASSO -> SVM -> KRR
Models = ["LinearRidge", "LinearLASSO", "LinearSVM", "KRR"];
Solvers = ["lbfgs","sparsa","dual","lbfgs"];
Regularisation = ["ridge","lasso","ridge","ridge"];
lambdas = logspace(-6,1,300);

for phenotype = 1:4
    if phenotype == 2
        Learners = ["logistic","logistic","svm","logistic"];
        MdlType = @fitclinear;
        ACCfun = @(x,y) mean(x==y)*100;
    else
        Learners = ["leastsquare","leastsquare","svm","leastsquare"];
        MdlType = @fitrlinear;
        ACCfun = @(x,y) mean(abs(x-y));
        Corrfun = @(x,y) corr(x,y);
    end
    TargetNumeric = TargetVariables(:,phenotype);
    for modelchoice = 1:4
        ModeltoTrain = @(x,y) MdlType(x,y,'ObservationsIn','columns',...
            'Learner',Learners(modelchoice),'Solver',Solvers(modelchoice),'Lambda',lambdas,'Regularization',Regularisation(modelchoice));
        ResultTable = table();
        for weight = 1:length(Weights)
            RunClassicalMethods
            ResultTable.(Weights(weight)) = resultcolumn;
            AllBetas.(Models(modelchoice)).(Phenotypes(phenotype)).(Weights(weight)) = GetBetas;
        end
        AllResults.(Models(modelchoice)).(strcat(Phenotypes(phenotype),"_PredictionPerformanceTable")) = ResultTable;
        save(fullfile(currentpath,'MLResultsForAllPredictionTasks.mat'),'AllResults');
        save(fullfile(currentpath,'MLBetasForAllPredictionTasks.mat'),'AllBetas');
    end
end
