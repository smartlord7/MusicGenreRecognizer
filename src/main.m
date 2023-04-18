% Clear figures and workspace variables
close all
clear

% Make this program reproducible
rng(0)

% Constants
DIRECTORY_DATA = "../data/";
EXTENSION_FEATURES = ".csv";
EXTENSION_IMG = ".png";
PATH_PLOT_IMAGES = "../img/plots/";
PATH_FEATURES = DIRECTORY_DATA + "features" + EXTENSION_FEATURES;
FRACTION_DEVELOPMENT = 0.8;
FRACTION_TESTING = 0.2;
FRACTION_TRAINING = FRACTION_DEVELOPMENT * 0.8;
FRACTION_VALIDATION = FRACTION_DEVELOPMENT * 0.2;
FUNCTIONS_NORMALIZATION = ["zscore", "norm", "range"];
FUNCTIONS_DISTANCES = ["euclidean", "cityblock", "minkowski", "chebychev", "mahalanobis"];
N_PROJECTION_FEATURES = 15;
N_TOP_DISCRIMINANT_KW_RANKED_FEATURES = 40;
N_TOP_DISCRIMINANT_RF_RANKED_FEATURES = 25;

% Load data
fprintf("Loading dataset...\n");
csv_data = readtable(PATH_FEATURES);
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
features = to_data_struct(features, target_num);

results_table = cell(size(FUNCTIONS_NORMALIZATION, 2) * ...
    size(FUNCTIONS_DISTANCES, 2) * ...
    size(target_labels, 2), 8);

count = 1;
metadata = struct;
metadata.n_top_features_kw = N_TOP_DISCRIMINANT_KW_RANKED_FEATURES;
metadata.n_top_features_rf = N_TOP_DISCRIMINANT_RF_RANKED_FEATURES;
metadata.dim_pca = N_PROJECTION_FEATURES;
metadata.base_path = PATH_PLOT_IMAGES;
metadata.ext = EXTENSION_IMG;
metdata.plot = true;

for i=(1:size(FUNCTIONS_NORMALIZATION, 2))
    norm_function = FUNCTIONS_NORMALIZATION(i);
    metadata.norm_function = norm_function;
    fprintf("Normalizing using '%s'...\n", norm_function);
    features_ = features;
    col_names_ = col_names;

    features_.X = normalize(features_.X, norm_function);

    features_ = rank_kruskal_wallis(features_, metadata);
    
    features_ = rank_random_forest(features_, metadata);

    features_ = project_pca(features_, metadata);
    

    % Minimum distance classifier


    data_mdc = features_; 
    [train_data, val_data, test_data] = divide_data(features_, FRACTION_TRAINING, ...
   FRACTION_VALIDATION,FRACTION_TESTING); % Divide the data for training, validation and testing
 

    for j=(1:size(target_labels))
        train_data_ = train_data;
        val_data_ = val_data;

        choice_class = j - 1;
        genre = target_labels{j};
        fprintf("Binary classification for genre: '%s'\n", genre);
        
        % Binarize target labels: 1 for the chosen class and 0 for the
        % remaining
        train_data_ = to_bin_classification(train_data_, choice_class);
        val_data_ = to_bin_classification(val_data_, choice_class);

        for k=(1:size(FUNCTIONS_DISTANCES, 2))
            dist_func = FUNCTIONS_DISTANCES(k);

            % Run the Minimum Distance Classifier
            predicted = min_dist_classifier(train_data_, val_data_, dist_func);
            file_path = PATH_PLOT_IMAGES + "cm_" + genre + "_" + norm_function + "_" + dist_func + EXTENSION_IMG;
            [mse, accuracy, specificity, sensitivity, f_measure] = eval_classifier(val_data_.y, predicted, file_path);
            fprintf("MSE: %.3f\n" + ...
                "Accuracy: %.3f\n" + ...
                "Specificity: %.3f\n" + ...
                "Sensitivity: %.3f\n" + ...
                "F-measure: %.3f\n", mse, accuracy, specificity, sensitivity, f_measure);
            
             % store the results in a cell array
                results_table{count, 1} = FUNCTIONS_NORMALIZATION(i);
                results_table{count, 2} = FUNCTIONS_DISTANCES(k);
                results_table{count, 3} = string(genre);
                results_table{count, 4} = mse;
                results_table{count, 5} = accuracy;
                results_table{count, 6} = specificity;
                results_table{count, 7} = sensitivity;
                results_table{count, 8} = f_measure;

                count = count + 1;        
        end

        % Reset labels
        data_mdc.y = features_.y;
    end
end

% convert the cell array to a table
header = {'Normalization function', 'Distance function', ...
'Class', 'MSE', 'Accuracy', 'Specificity', 'Sensitivity', 'F-measure'};
results_table = cell2table(results_table, 'VariableNames', header);

% write the table to an Excel file
writetable(results_table, 'results.xlsx');

% End of Minimum distance classifier

