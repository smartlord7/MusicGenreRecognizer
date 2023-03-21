function [y_predicted] = min_dist_classifier(data, dist_func)
%MDC_EUCLIDEAN Summary of this function goes here

% Separate the features and labels
x = data.X;
y = data.y;

% Find the class means
class_labels = unique(y);
n_classes = length(class_labels);

class_means = zeros(n_classes, data.dim);
for i = 1:n_classes
    class_means(i, :) = mean(x(:, y==class_labels(i)), 2)';
end

% Compute the euclidean distances
distances = pdist2(x', class_means, dist_func);
m = min(distances, [], 2);
idx = (distances == m);
[row_idx, col_idx] = find(idx == 1);
y_predicted = accumarray(row_idx, col_idx, [], @(x) sort(x)')';
y_predicted = y_predicted - 1; % labels start in 0

end

