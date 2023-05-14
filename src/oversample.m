function [oversampled_training_data, oversampled_training_labels] = oversample(training_data, training_labels)
% Oversamples the training data to balance the dataset
% Count the number of samples in each class
num_classes = max(training_labels);
num_samples_per_class = zeros(num_classes, 1);
for i = 1:num_classes
num_samples_per_class(i) = sum(training_labels == i);
end
% Determine the maximum number of samples in any class
max_num_samples = max(num_samples_per_class);
% Oversample the training data for each class
oversampled_training_data = [];
oversampled_training_labels = [];
for i = 1:num_classes
class_indices = find(training_labels == i);
num_samples = num_samples_per_class(i);
if num_samples < max_num_samples
oversampled_indices = randsample(class_indices, max_num_samples - num_samples, true);
oversampled_data = interp1(class_indices, training_data(class_indices, :), oversampled_indices, 'spline');
oversampled_labels = ones(max_num_samples - num_samples, 1) * i;
oversampled_training_data = [oversampled_training_data; oversampled_data];
oversampled_training_labels = [oversampled_training_labels; oversampled_labels];
end
end
% Concatenate the oversampled data with the original data
oversampled_training_data = [training_data; oversampled_training_data];
oversampled_training_labels = [training_labels; oversampled_training_labels];
% Shuffle the oversampled data
shuffled_indices = randperm(length(oversampled_training_labels));
oversampled_training_data = oversampled_training_data(shuffled_indices, :);
oversampled_training_labels = oversampled_training_labels(shuffled_indices);
end