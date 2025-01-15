function net1 = replaceLayersBetweenNets(net1, net2, layerType)
% Replace layers in the layerGraph lgraph of the type specified by
% layerType with copies of the layer newLayer.

for ilayer=1:length(net1.Layers)
    if isa(net1.Layers(ilayer), layerType)
        % Match names between old and new layer.
        layerName = net1.Layers(ilayer).Name;
        
        net1 = replaceLayer(net1, layerName, net2.Layers(ilayer));
    end
end
end