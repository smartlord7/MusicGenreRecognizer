function [data] = min_max_norm(data)

% Perform min-max normalization on the 'X' field
data.X = normalize(data.X, 'range');

end

