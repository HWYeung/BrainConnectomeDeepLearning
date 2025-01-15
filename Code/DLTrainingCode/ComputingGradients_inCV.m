options = trainingOptions('sgdm', 'InitialLearnRate',eps, 'ResetInputNormalization',false,'OutputFcn',@(~)true,'Verbose',false );
net = trainNetwork(X,Y,layerGraph(net) ,options);
lgraph1 = layerGraph(net); % Continue from Accuracy_Epoch_Plot.m
lgraph1 = removeLayers(lgraph1,lgraph1.Layers(end).Name);
if phenotype == 2
    lgraph1 = removeLayers(lgraph1,lgraph1.Layers(end).Name);
end
softmaxName = lgraph1.Layers(end,1).Name;
dlnet = dlnetwork(lgraph1);
dydI = dlarray(zeros(85,85,1,size(X,4)));
for image = 1:size(X,4)
    dlImg = dlarray(X(:,:,:,image),'SSC');
    dydI_temp1 = dlfeval(@gradientMap,dlnet, dlImg, softmaxName, 1);
    dydI_temp = dydI_temp1(:,:,1).*(dlImg(:,:,1)>0);
    dydI(:,:,:,image) = dydI_temp;%./max(max(abs(dydI_temp)));
end
D = squeeze(extractdata(dydI));
meanGrad1 = mean(D,3);
MeanGrad = meanGrad1 + meanGrad1';
clear D dydI meanGrad1 lgraph1
