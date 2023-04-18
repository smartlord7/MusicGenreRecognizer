function [proj, eigenValues, eigenVectors, individual_variance] = project_pca_(data, dim)
% Performs PCA on the input dataset in the 'data' struct
% Returns the principal components (eigenVectors), the transformed data (proj), and the eigenvalues (eigenValues)
    x = data.X;
    data.X = x;
    pcaModel = pca(x, dim);
    eigenValues = pcaModel.eigval;
    eigenVectors = pcaModel.W;
    proj = linproj(data, pcaModel);
    total_variance = sum(eigenValues);
    individual_variance = cumsum(eigenValues) ./ total_variance * 100;
end



