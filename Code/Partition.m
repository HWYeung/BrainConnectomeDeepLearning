function [partition] = Partition(participants,folds)
testpartition = cell(1,folds);
testnumber = round(participants/folds);
for i = 1:folds-1
    testpartition{i}=(i-1)*testnumber+1:i*testnumber;
end
testpartition{folds}=(folds-1)*testnumber+1:participants;
trainingpartition = cellfun(@(x) setdiff(1:participants,x),testpartition,'UniformOutput',false);
partition = [trainingpartition;testpartition];
end
