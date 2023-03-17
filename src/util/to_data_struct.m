function [data_struct] = to_data_struct(x, y, varargin)
    data_struct.X = x';
    data_struct.y = y';
    data_struct.dim = size(data_struct.X, 1);
    data_struct.num_data = size(data_struct.X, 2);
        
    % Optional argument: dataset name
    if nargin == 3   
        data_struct.name = varargin(3);
    end
end
