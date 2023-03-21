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
N_PROJECTION_FEATURES = 25;
N_TOP_DISCRIMINANT_KW_RANKED_FEATURES = 15;
N_TOP_DISCRIMINANT_RF_RANKED_FEATURES = 10;

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

% PCA
results_table = cell(size(FUNCTIONS_NORMALIZATION, 2) * ...
    size(FUNCTIONS_DISTANCES, 2) * ...
    size(target_labels, 2), 8);

count = 1;
for i=(1:size(FUNCTIONS_NORMALIZATION, 2))
    norm_function = FUNCTIONS_NORMALIZATION(i);
    fprintf("Normalizing using '%s'...\n", norm_function);
    features_ = features;
    col_names_ = col_names;

    features_.X = normalize(features_.X, norm_function);
    x = features_.X;
    fprintf("Applying PCA for %d dimensions...\n", N_PROJECTION_FEATURES);
    [proj, eigenValues, eigenVectors, individual_variance] = project_pca(features_, N_PROJECTION_FEATURES);
    features_.X = proj.X;
    features_.dim = size(proj.X, 1);
    

    figure;
    plot(eigenValues, 'o-.');
    xlabel('Principal component');
    ylabel('Eigen value');
    file_path = PATH_PLOT_IMAGES + "pca_eigen_values_" + norm_function + EXTENSION_IMG;
    save_img(file_path);
    
    figure;
    plot(individual_variance, 'o-');
    xlabel('Principal component');
    ylabel('% of variance');
    file_path = PATH_PLOT_IMAGES + "pca_variance_" + norm_function + EXTENSION_IMG;
    save_img(file_path);

    % Correlation study
    
    fprintf("Calculating %d PCA features correlation matrix...\n", N_PROJECTION_FEATURES);
    corr_matrix = corrcoef(features_.X'); % Only the ones most important in PCA analysis
    
    figure;
    heatmap((1:N_PROJECTION_FEATURES), (1:N_PROJECTION_FEATURES), corr_matrix);
    file_path = PATH_PLOT_IMAGES + "corr_matrix_pca_" + norm_function + EXTENSION_IMG;
    save_img(file_path);

    figure;
    ppatterns(features_);
    file_path = PATH_PLOT_IMAGES + "patterns_pca" + EXTENSION_IMG;
    save_img(file_path);

    % Kruskal Wallis

    [features_h, features_idx] = rank_kruskal_wallis(features_);
    top_features = features_h(1:N_TOP_DISCRIMINANT_KW_RANKED_FEATURES);
    top_features_idx = features_idx(1:N_TOP_DISCRIMINANT_KW_RANKED_FEATURES);
    
    fprintf("Applying Kruskal-Wallis test...\n");
    fprintf("-------Top %d KW ranked features-------\n", N_TOP_DISCRIMINANT_KW_RANKED_FEATURES);

    for j=(1:N_TOP_DISCRIMINANT_KW_RANKED_FEATURES)
        fprintf("%d - %d - H = %.3f\n", j, features_idx(j), features_h(j));
    end
    
    features_.X = features_.X(top_features_idx, :); % Update the features to the ones most relevant
    features_.dim = size(features_.X, 1);
%     test_norm(features_, col_names_);
    
    figure;
    scatter((1:size(features_h, 2)), features_h);
    file_path = PATH_PLOT_IMAGES + "scatter_h_kw" + EXTENSION_IMG;
    save_img(file_path);

    figure;
    ppatterns(features_);
    file_path = PATH_PLOT_IMAGES + "patterns_pca_kw" + EXTENSION_IMG;
    save_img(file_path);

    
    % End of Kruscal Wallis
    
    % Correlation study
    
    fprintf("Calculating %d top KW features correlation matrix...\n", N_TOP_DISCRIMINANT_KW_RANKED_FEATURES);
    corr_matrix = corrcoef(features_.X'); % Only 20 best
    figure;
    heatmap((1:features_.dim), (1:features_.dim), corr_matrix);
    file_path = PATH_PLOT_IMAGES + "corr_matrix_kw" + EXTENSION_IMG;
    save_img(file_path);
    
    % End of correlation study
    
    % RandomForest
    
    importances = rank_rf_importance(features_);
    [importances, idx] = sort(importances, "descend");
    idx = idx(1:N_TOP_DISCRIMINANT_RF_RANKED_FEATURES);
    features_.X = features_.X(idx, :);
    features_.dim = size(features_.X, 1);

    fprintf("Applying Random Forest test...\n");
    fprintf("-------Top %d RF ranked features by importance-------\n", N_TOP_DISCRIMINANT_RF_RANKED_FEATURES);

    for j=(1:N_TOP_DISCRIMINANT_RF_RANKED_FEATURES)
        fprintf("%d - %d - I = %.3f\n", j, idx(j), importances(j));
    end
    
    file_path = PATH_PLOT_IMAGES + "importance_rf" + EXTENSION_IMG;
    save_img(file_path);
    
    % End of RandomForest

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

% End of PCA

% End of Minimum distance classifier



