%% Create Deep Learning Network Architecture
% Script for creating the layers for a deep learning network with:

%% Create the Layer Graph
% Create the layer graph variable to contain the network's layers.

lgraph = layerGraph();
%% Add the Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.
M = 85;
%"BiasLearnRateFactor",0,"WeightL2Factor",0,"WeightLearnRateFactor",0
%    crossChannelNormalizationLayer(3,'K',1,"Name","crosschannel_1")
tempLayers = [imageInputLayer([M M 3],"Name","imageinput","Normalization","none")
    mySplitLayer(3, "split")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalMaxPooling2dLayer("Name","gmaxpool1");
lgraph = addLayers(lgraph,tempLayers);
tempLayers = globalMaxPooling2dLayer("Name","gmaxpool2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 M],8,"Name","conv_1")
    RepmatLayerCol('RepmatCol_1')];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([M 1],8,"Name","conv_2")
    RepmatLayerRow('RepmatRow_1')];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    batchNormalizationLayer
    leakyReluLayer(0.2,'Name',"clippedleaky1")
    dropoutLayer(0.3,"Name","dropout_1")
    ];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 M],8,"Name","conv_3")
    RepmatLayerCol('RepmatCol_2')];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([M 1],8,"Name","conv_4")
    RepmatLayerRow('RepmatRow_2')];
lgraph = addLayers(lgraph,tempLayers);


tempLayers = [
    additionLayer(2,"Name","addition_2")
    batchNormalizationLayer
    leakyReluLayer(0.2,"Name", "clippedleaky2")
    dropoutLayer(0.3,"Name","dropout_2")
    ];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = convolution2dLayer([1 M],4,"Name","conv_5");
lgraph = addLayers(lgraph,tempLayers);
tempLayers = convolution2dLayer([M 1],4,"Name","conv_6");
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    TransposeAdd("transadd") 
    batchNormalizationLayer
    leakyReluLayer(0.2,"Name","lrelu_4")
    dropoutLayer(0.3,"Name","dropout_7")
    fullyConnectedLayer(30,"Name","fc_1a")
    batchNormalizationLayer
    leakyReluLayer(0.2,"Name","lrelu_5")
    dropoutLayer(0.3,"Name","dropout_4")
    ];

lgraph = addLayers(lgraph,tempLayers);
if phenotype == 2 % for Sex Classification
tempLayers = [
    concatenationLayer(3,3,'Name',"concat")
    fullyConnectedLayer(2,"Name","fc_4")
    softmaxLayer("Name","softmax")
    classificationLayer('Name','output')];
else
tempLayers = [
    concatenationLayer(3,3,'Name',"concat")
    fullyConnectedLayer(1,"Name","fc_4")
    regressionLayer('Name','routput')];
end
lgraph = addLayers(lgraph,tempLayers);
%% Connect the Layer Branches
% Connect all the branches of the network to create the network's graph.

lgraph = connectLayers(lgraph,"split/out1","conv_1");
lgraph = connectLayers(lgraph,"split/out1","conv_2");
lgraph = connectLayers(lgraph,"split/out2","gmaxpool1");
lgraph = connectLayers(lgraph,"split/out3","gmaxpool2");
lgraph = connectLayers(lgraph,'RepmatCol_1',"addition_1/in1");
lgraph = connectLayers(lgraph,'RepmatRow_1',"addition_1/in2");
lgraph = connectLayers(lgraph,"dropout_1","conv_3");
lgraph = connectLayers(lgraph,"dropout_1","conv_4");
lgraph = connectLayers(lgraph,'RepmatCol_2',"addition_2/in2");
lgraph = connectLayers(lgraph,'RepmatRow_2',"addition_2/in1");
lgraph = connectLayers(lgraph,"dropout_2","conv_5");
lgraph = connectLayers(lgraph,"dropout_2","conv_6");
lgraph = connectLayers(lgraph,"conv_5","transadd/in1");
lgraph = connectLayers(lgraph,"conv_6","transadd/in2");
lgraph = connectLayers(lgraph,"dropout_4","concat/in1");
lgraph = connectLayers(lgraph,"gmaxpool1","concat/in2");
lgraph = connectLayers(lgraph,"gmaxpool2","concat/in3");
%% Clean Up Helper Variable

clear tempLayers;
%% Plot the Layers