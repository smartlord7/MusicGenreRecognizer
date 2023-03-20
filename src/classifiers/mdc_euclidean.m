function [rate, errors] = mdc_euclidean(data)
%MDC_EUCLIDEAN Summary of this function goes here

% Separate the features and labels
train_X = data.X;
train_Y = data.y;

% Find the class means
class_labels = unique(train_Y);
n_samples = size(train_X, 2);
n_classes = length(class_labels);

class_means = zeros(n_classes, n_samples);
for i = 1:n_classes
    class_means(i,:) = mean(train_X(train_Y==class_labels(i),:));
end

% Compute the euclidean distances
distances = pdist2(train_X, class_means, 'euclidean');

% Preallocate errors
errors = zeros(1, n_classes);

% Classify the samples
for i = 1:n_samples
    idx = find(distances(i, :) == min(distances(i, :)));
    errors(labels(i)) = errors(labels(i)) + (idx ~= labels(i));
end

% Compute the total error rate
rate = sum(errors) / n_samples;

end

