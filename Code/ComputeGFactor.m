function Gfactor = ComputeGFactor(CognitiveTasks, TestSetIndex)
%Here the Cognitive Tasks values are transformed values, transformation: refer to Yeung et al. 2022
% Predicting sex, age, general cognition and mental health with machine learning on brain structural connectomes
% https://doi.org/10.1002/hbm.26182

% observations in rows
sz = size(CognitiveTasks,1);
Gfactor = nan*ones(sz,1);
I = sum(isnan(CognitiveTasks),2) > 0;
CognitiveTasks = CognitiveTasks(~I,:);

if nargin < 2
    TestSetIndex = 1:1230;
end
L = size(CognitiveTasks,1);
TrainIndex = setdiff(1:L,TestSetIndex);
CognitiveTasksTrain = CognitiveTasks(TrainIndex,:);
Mean = nanmean(CognitiveTasksTrain);
Std = nanstd(CognitiveTasksTrain);
NormedCognitiveTasksTrain = (CognitiveTasksTrain - Mean)./Std;
corrp = corr(CognitiveTasksTrain,'Rows','pairwise');
[COEFF,~,~] = pcacov(corrp);
stdG = nanstd(NormedCognitiveTasksTrain*COEFF(:,1));
NormedCognitiveTasks = (CognitiveTasks - Mean)./Std;
Gfactor(~I) = (NormedCognitiveTasks*COEFF(:,1))./stdG;


