
X1 = Connectome.(Weights(weight));
Y = TargetNumeric;
Index = ~isnan(Y);
Y = Y(Index);
L = sum(Index);
TestI = 1:1230;
I = 1231:L;
if phenotype == 4
    TestI = 1:940;
    I = 941:L;
end
[partition] = Partition(length(I),5);

    
[DataMat, ~] = ConnectomeToDataMat(X1(:,:,Index));
GetBetas = zeros(size(DataMat,1),5);
clear X1
if phenotype > 2
    Sex = GetCovar(Index,2);
    Sex(Sex == 0) = -1;
    Age = GetCovar(Index,1);
    normedAge = (Age - mean(Age(I)))./ std(Age(I));
    DataMat = [Sex' ; Age' ; DataMat];
end
XTest = DataMat(:,TestI);
YTest = Y(TestI);
if phenotype == 2
    results = zeros(3,5);
else
    results = zeros(6,5);
end
disp(strcat("Training Model: ",Models(modelchoice),", for phenotype: ",Phenotypes(phenotype),", with Connectome weight: ",Weights(weight)));
for cv = 1:5   
    TrainI = I(partition{1,cv}); 
    ValidI = I(partition{2,cv});
    XTrain = DataMat(:,TrainI);
    YTrain = Y(TrainI);
    XValid = DataMat(:,ValidI);
    YValid = Y(ValidI);

    if modelchoice == 4
        if phenotype > 2
            Transform = @(x) [x(1:2,:);corr(XTrain,x)];
        else
            Transform = @(x) corr(XTrain,x);
        end
    else
        Transform = @(x) x;
    end



    Mdl = ModeltoTrain(Transform(XTrain),YTrain);

    validation = predict(Mdl,Transform(XValid),'ObservationsIn','columns');
    train = predict(Mdl,Transform(XTrain),'ObservationsIn','columns');
    test = predict(Mdl,Transform(XTest),'ObservationsIn','columns');
    ACCTrain = ACCfun(train,YTrain);
    ACCValid = ACCfun(validation,YValid);
    ACCTest = ACCfun(test,YTest);
    if phenotype ~= 2
        CorrTrain = corr(train,YTrain);
        CorrValid = corr(validation,YValid);
        CorrTest = corr(test,YTest);
    end

    if phenotype == 2
        Cost = ACCValid.*(ACCTrain > ACCValid);
        maxValidate = find(ACCValid == max(Cost));
    else
        Cost = CorrValid.*(CorrTrain > CorrValid).*(ACCTrain' < ACCValid');
        maxValidate = find(CorrValid == max(Cost));
    end

    if isempty(maxValidate)
        OptimPosition = 1;
    else
        OptimPosition = maxValidate(1);
    end
    results(1,cv) = ACCTrain( OptimPosition);
    results(2,cv) = ACCValid( OptimPosition);
    results(3,cv) = ACCTest(OptimPosition);
    if phenotype ~=2
        results(4,cv) = CorrTrain( OptimPosition);
        results(5,cv) = CorrValid( OptimPosition);
        results(6,cv) = CorrTest( OptimPosition);
    end


    if modelchoice ~= 4
        if phenotype > 2
            GetBetas(:,cv) = Mdl.Beta(3:end,OptimPosition);
        else
            GetBetas(:,cv) = Mdl.Beta(:,OptimPosition);
        end
    end
    clear train test validation
end

clear DataMat
if size(results,1) == 3
    resultcolumn = [mean(results,2);std(results,[],2)];
else
    resultcolumn = mean(results,2);
end