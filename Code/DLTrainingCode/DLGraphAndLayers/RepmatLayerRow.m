classdef RepmatLayerRow < nnet.layer.Layer
    % Example custom ElementWiseMultiplication layer.
    properties (Learnable)
        % Layer learnable parameters
            
        % Scaling coefficients
    end
    
    methods
        function layer = RepmatLayerRow(name) 
            % layer = ElementWiseMultiplication(numInputs,name) creates a
            % element wise multiplication and specifies the number of inputs
            % and the layer name.
            % Set number of inputs.
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "Repmat Layer of inputs";
        
        end
        
        function Z = predict(~, X1)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.     
            % Element Wise Multiplication
                        sz = size(X1);
                        B = ones([sz(2) 1 1 1],'like',X1);
                        Z = B.*X1;
        
        end
    end
end