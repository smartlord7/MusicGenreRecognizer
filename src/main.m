PATH_DATA = "../data/";


close all
clear

% Load data
csv_data = readtable('dados.csv');

col_names = csv_data.Properties.VariableNames;

% Define a cell array of the target labels
target_labels = {'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'};

% Use the ismember function to convert the target labels to numerical format
target_num = zeros(size(csv_data, 1), 1);
for i = 1:numel(target_labels)
    target_num(ismember(csv_data.label, target_labels{i})) = i;
end

% Replace the 'target' column in the table with the numerical values
csv_data.label = target_num;

data_full = table2array(csv_data(:, 2:end));

ix=randperm(size(data_full,1)); %randomize index

ixdataset=ix(1:floor(800)); %dataset index
ixvalidation=ix(floor(800)+1:end); %validationset index

ixtraining=ixdataset(1:floor(650)); %training index
ixtesting=ixdataset(floor(650)+1:end); %testing index