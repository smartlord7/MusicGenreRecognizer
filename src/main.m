% Clear figures and workspace variables
close all
clear

% Constants
DIRECTORY_DATA = "../data/";
EXTENSION_FEATURES = ".csv";
PATH_FEATURES = DIRECTORY_DATA + "features" + EXTENSION_FEATURES;
FRACTION_DEVELOPMENT = 0.8;
FRACTION_TESTING = 0.2;
FRACTION_TRAINING = 0.8;
FRACTION_VALIDATION = 0.2;
FUNCTIONS_NORMALIZATION = ["zscore", "norm", "range"];
N_TOP_RANKED_FEATURES = 20;

% Load data
csv_data = readtable(PATH_FEATURES);
col_names = csv_data.Properties.VariableNames(2:end - 1);

% Define a cell array of the target labels
target_labels = table2cell(unique(csv_data(:, end)));

% Use the ismember function to convert the target labels to numerical format
target_num = zeros(size(csv_data, 1), 1);
for i = 1:numel(target_labels)
    target_num(ismember(csv_data.label, target_labels{i})) = i;
end

features = table2array(csv_data(:, 2:end - 1));
features = to_data_struct(features, target_num);
[features_h, features_idx] = rank_kruskal_wallis(features);

top_n_features_h = features_h(1:N_TOP_RANKED_FEATURES + 1);
fprintf("-------Top %d KW ranked features-------\n", N_TOP_RANKED_FEATURES)
for i=(1:N_TOP_RANKED_FEATURES)
    fprintf("%d - %s - H = %.3f\n", i, col_names{features_idx(i)}, features_h(i));
end

figure;
plot( (1:features.dim), features_h);

%[train_data, val_data, test_data] = divide_data(features, FRACTION_TRAINING, ...
%    FRACTION_VALIDATION,FRACTION_TESTING); % Divide the data for training, validation and testing



