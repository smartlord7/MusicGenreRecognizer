function [coeff, score, latent] = pca_data(data)
% Performs PCA on the input dataset in the 'data' struct
% Returns the principal components (coeff), the transformed data (score), and the eigenvalues (latent)

% Extract the input features
X = data.X;

% Standardize the input data
X = zscore(X);

% Perform PCA
[coeff, score, latent] = pca(X);


