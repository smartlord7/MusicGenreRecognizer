function [features] = rank_kruskal_wallis(features, metadata)
    n_top_features = metadata.n_top_features_kw;
    base_path = metadata.base_path;
    ext = metadata.ext;

    % Kruskal Wallis

    [features_h, features_idx] = rank_kruskal_wallis_(features);
    top_features_idx = features_idx(1:n_top_features);
    
    fprintf("Applying Kruskal-Wallis test...\n");
    fprintf("-------Top %d KW ranked features-------\n", n_top_features);

    for j=(1:n_top_features)
        fprintf("%d - %d - H = %.3f\n", j, features_idx(j), features_h(j));
    end
    
    features.X = features.X(top_features_idx, :); % Update the features to the ones most relevant
    features.dim = size(features.X, 1);
    % test_norm(features, col_names_);
    
    figure;
    scatter((1:size(features_h, 2)), features_h);
    file_path = base_path + "scatter_h_kw" + ext;
    save_img(file_path);

    figure;
    ppatterns(features);
    file_path = base_path + "patterns_pca_kw" + ext;
    save_img(file_path);

    % End of Kruscal Wallis
    
    % Correlation study
    
    fprintf("Calculating %d top KW features correlation matrix...\n", n_top_features);
    corr_matrix = corrcoef(features.X'); % Only 20 best
    figure;
    heatmap((1:features.dim), (1:features.dim), corr_matrix);
    file_path = base_path + "corr_matrix_kw" + ext;
    save_img(file_path);
    
    % End of correlation study
end

