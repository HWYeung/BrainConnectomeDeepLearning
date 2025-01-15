function Pfactor = ComputePFactor(MHQItems, TestSetIndex)
% Refer to Yeung et al. 2022 for the 14 MHQ items 
% Predicting sex, age, general cognition and mental health with machine learning on brain structural connectomes
% https://doi.org/10.1002/hbm.26182

% observations in rows
sz = size(MHQItems,1);
Pfactor = nan*ones(sz,1);
I = sum(isnan(MHQItems),2) > 0;
MHQItems = MHQItems(~I,:);
if nargin < 2
    TestSetIndex = 1:940;
end
L = size(MHQItems,1);
TrainIndex = setdiff(1:L,TestSetIndex);
MHQItemsTrain = MHQItems(TrainIndex,:);
Mean = nanmean(MHQItemsTrain);
Std = nanstd(MHQItemsTrain);
NormedMHQItemsTrain = (MHQItemsTrain - Mean)./Std;
PolyCorr = PolychoricCorr(MHQItemsTrain);
[COEFF,~,~] = pcacov(PolyCorr);
stdP = nanstd(NormedMHQItemsTrain*COEFF(:,1));
NormedMHQItems = (MHQItems - Mean)./Std;
Pfactor(~I) = (NormedMHQItems*COEFF(:,1))./stdP;


