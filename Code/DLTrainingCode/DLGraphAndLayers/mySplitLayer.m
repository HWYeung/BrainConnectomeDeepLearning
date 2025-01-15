classdef mySplitLayer < nnet.layer.Layer
    % Custom softmax layer.
    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end
    methods
        function layer = mySplitLayer(outputs, name)
            % layer = mySoftmaxLayer(name) creates a layer
            % and specifies the layer name.
            layer.NumOutputs = outputs;
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "My softmax layer";
        end
        
        function varargout = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            varargout = {};
            sz = size(X);
            for i = 1:sz(3)
                varargout{i} = X(:,:,i,:);
            end
        end
        
    end
end