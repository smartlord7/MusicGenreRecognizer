function [X_resampled, Y_resampled] = smote(X, Y, k, N)

% X - input data
% Y - target labels
% k - number of nearest neighbors to use for SMOTE
% N - number of synthetic samples to generate for each minority sample

% Split the data into minority and majority classes
X_minority = X(Y == 1, :);
Y_minority = Y(Y == 1);

X_majority = X(Y ~= 1, :);
Y_majority = Y(Y ~= 1);

% Calculate the number of minority samples and the number of synthetic samples to generate
num_minority = size(X_minority, 1);
num_synthetic = N * num_minority;

% Compute the k-nearest neighbors for each minority sample
knn_indices = knnsearch(X_majority, X_minority, 'K', k);

% Generate synthetic samples
X_synthetic = zeros(num_synthetic, size(X, 2));
Y_synthetic = ones(num_synthetic, 1);

synthetic_index = 1;
for i = 1:num_minority
    % Get the k-nearest neighbors for the current minority sample
    neighbors = X_majority(knn_indices(i, :), :);

    % Generate N synthetic samples for the current minority sample
    for j = 1:N
        % Randomly select one of the k-nearest neighbors
        neighbor_index = knn_indices(i, randi(k));

        % Generate a synthetic sample
        diff = neighbors(neighbor_index, :) - X_minority(i, :);
        factor = rand;
        X_synthetic(synthetic_index, :) = X_minority(i, :) + factor * diff;

        % Update the synthetic sample label
        Y_synthetic(synthetic_index) = 1;

        % Increment the synthetic sample index
        synthetic_index = synthetic_index + 1;
    end
end

% Combine the original and synthetic samples
X_resampled = [X_majority; X_minority; X_synthetic];
Y_resampled = [Y_majority; Y_minority; Y_synthetic];

end