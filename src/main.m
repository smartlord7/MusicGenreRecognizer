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


% Load data
csv_data = readtable(PATH_FEATURES);
col_names = csv_data.Properties.VariableNames;

% Define a cell array of the target labels
target_labels = table2cell(unique(csv_data(:, end)));

% Use the ismember function to convert the target labels to numerical format
target_num = zeros(size(csv_data, 1), 1);
for i = 1:numel(target_labels)
    target_num(ismember(csv_data.label, target_labels{i})) = i;
end

% Replace the 'target' column in the table with the numerical values
csv_data.label = target_num;

data_full = table2array(csv_data(:, 2:end));

[train_data, val_data, test_data] = divide_data(data_full, FRACTION_TRAINING, ...
    FRACTION_VALIDATION,FRACTION_TESTING); % Divide the data for training, validation and testing


