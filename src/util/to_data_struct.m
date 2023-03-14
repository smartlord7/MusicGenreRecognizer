function data_struct = to_data_struct(dataX, dataY, varargin)
    
    if nargin == 2
        % Code to handle case where no optional argument is provided
        data_struct.X=dataX';
        data_struct.y=dataY';
        data_struct.dim=size(data_struct.X,1);
        data_struct.num_data=size(data_struct.X,2);
        
    elseif nargin == 3
        % Code to handle case where optional argument is provided
        data_struct.X=dataX';
        data_struct.y=dataY';
        data_struct.dim=size(data_struct.X,1);
        data_struct.num_data=size(data_struct.X,2);
        data_struct.name=varargin;
    else
        % Error message for invalid number of input arguments
        error('Invalid number of input arguments');
    end












    
 
end

