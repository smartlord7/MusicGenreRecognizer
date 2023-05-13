function [features] = project_pca(features, metadata)
    norm_function = metadata.norm_function;
    dim = metadata.dim_pca;
    base_path = metadata.base_path;
    ext = metadata.ext;

    % PCA

    fprintf("Applying PCA for %d dimensions...\n", dim);
    [proj, eigenValues, ~, individual_variance] = project_pca_(features, dim);
    features.X = proj.X;
    features.dim = size(proj.X, 1);

    if Const.PLOT == true
        figure;
        plot(eigenValues, 'o-.');
        xlabel('Principal component');
        ylabel('Eigen value');
        file_path = base_path + "pca_eigen_values_" + norm_function + ext;
        save_img(file_path);
        
        figure;
        plot(individual_variance, 'o-');
        xlabel('Principal component');
        ylabel('% of variance');
        file_path = base_path + "pca_variance_" + norm_function + ext;
        save_img(file_path);

    end
    % End of PCA

    % Correlation study
    
    fprintf("Calculating %d PCA features correlation matrix...\n", dim);
    corr_matrix = corrcoef(features.X'); % Only the ones most important in PCA analysis
    
    if Const.PLOT == true
        figure;
        heatmap((1:dim), (1:dim), corr_matrix);
        file_path = base_path + "corr_matrix_pca_" + norm_function + ext;
        save_img(file_path);
    
        figure;
        ppatterns(features);
        file_path = base_path + "patterns_pca" + ext;
        save_img(file_path);
    end

    % End of correlation study
end

