function [DataMat, IndexMat] = ConnectomeToDataMat(ConnectomeMat)
% Flattening the uppertriangular entries of connectomes for classical ML
% methods
sz = size(ConnectomeMat,3);
feat = numel(squareform(ConnectomeMat(:,:,1) - diag(diag(ConnectomeMat(:,:,1)))));
DataMatT = zeros(sz,feat);
for parti = 1:sz
    DataMatT(parti,:) = squareform(ConnectomeMat(:,:,parti) - diag(diag(ConnectomeMat(:,:,parti))));
end
Null = find(sum(DataMatT)==0);
IndexMat = zeros(feat,feat - numel(Null));
HasFeat = setdiff(1:feat,Null);
for feature = 1:numel(HasFeat)
    IndexMat(HasFeat(feature),feature) = 1;
end
DataMatT2 = DataMatT*IndexMat;
DataMat = DataMatT2';
end