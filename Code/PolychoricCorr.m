function [PolyCorr,Corr] = PolychoricCorr(X)
IterationNumber = 10*floor(size(X,1).^(1/2));
SamplesNum = floor(size(X,1)/2);
Corr = zeros(size(X,2),size(X,2),IterationNumber);
for iterationsampling = 1:IterationNumber
    Samples = randsample(1:size(X,1),SamplesNum);
    Corr(:,:,iterationsampling) = polychoric_proc_missing(X(Samples,:),999);
end
PolyCorr = median(Corr,3);