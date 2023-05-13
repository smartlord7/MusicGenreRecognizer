function [col_names, features, target_labels] = import_dataset(path, name)
    % Load data
    fprintf("Loading dataset...\n");
    csv_data = readtable(path);
    col_names = csv_data.Properties.VariableNames(2:end - 1);
    
    for i=(1:size(col_names, 2))
        rep = strrep(col_names{i}, '_', '-');
        col_names{i} = rep; % Replace underscores to prevent bad format in plots
    end
    
    % Define a cell array of the target labels
    target_labels = table2cell(unique(csv_data(:, end))); % Discard first column (file name) and last column (label)
    
    % Use the ismember function to convert the target labels to numerical format
    target_num = zeros(size(csv_data, 1), 1);
    for i = 1:numel(target_labels)
        target_num(ismember(csv_data.label, target_labels{i})) = i - 1;
    end
    
    features = table2array(csv_data(:, 2:end - 1));
    features = to_data_struct(features, target_num, name);
end

