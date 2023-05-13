function [y_predicted_train, y_predicted_val] = min_dist_classifier(train_data, val_data, dist_func, classif_type)
    %MDC_EUCLIDEAN Summary of this function goes here
    
    % Separate the features and labels
    x = train_data.X;
    y = train_data.y;
    
    % Find the class means
    class_labels = unique(y);
    n_classes = length(class_labels);
    
    class_means = zeros(n_classes, train_data.dim);
    for i = 1:n_classes
        class_means(i, :) = mean(x(:, y==class_labels(i)), 2)';
    end
    
    % Compute the euclidean distances
    x_val = val_data.X;
    distances_val = pdist2(x_val', class_means, dist_func);
    distances_train = pdist2(train_data.X', class_means, dist_func);
    
    
    if (classif_type == "Binary") 
    
        m = min(distances_train, [], 2);
        idx = (distances_train == m);
        [row_idx, col_idx] = find(idx == 1);
        y_predicted_train = accumarray(row_idx, col_idx, [], @(x) sort(x)')';
        y_predicted_train = y_predicted_train - 1; % labels start in 0
    
        m2 = min(distances_val, [], 2);
        idx2 = (distances_val == m2);
        [row_idx2, col_idx2] = find(idx2 == 1);
        y_predicted_val = accumarray(row_idx2, col_idx2, [], @(x) sort(x)')';
        y_predicted_val = y_predicted_val - 1; % labels start in 0
    
    else
        
        [m, min_idx_val] = min(distances_val, [], 2);
        [m, min_idx_train] = min(distances_train, [], 2);
        y_predicted_train = class_labels(min_idx_train);
        y_predicted_val = class_labels(min_idx_val);
        
    end
end

