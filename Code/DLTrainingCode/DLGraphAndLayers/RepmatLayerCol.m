classdef RepmatLayerCol < nnet.layer.Layer
    % Example custom ElementWiseMultiplication layer.
    properties
            % (Optional) Layer properties.
        % Scaling coefficients
    end
    
    methods
        function layer = RepmatLayerCol(name) 
            % layer = ElementWiseMultiplication(numInputs,name) creates a
            % element wise multiplication and specifies the number of inputs
            % and the layer name.
            % Set number of inputs.
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "Repmat Layer of input";
        
        end
        
        function Z = predict(~, X1)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.     
            % Element Wise Multiplication
                        sz = size(X1);
                        B = ones([1 sz(1) 1 1],'like',X1);
                        Z = X1.*B;
                        
                            
        
        end
    end
end