% Clear figures and workspace variables
close all
clear

% Constants
DIRECTORY_DATA = "../data/";
EXTENSION_FEATURES = ".csv";
EXTENSION_IMG = ".png";
PATH_PLOT_IMAGES = "../img/plots/";
PATH_FEATURES = DIRECTORY_DATA + "features" + EXTENSION_FEATURES;
FRACTION_DEVELOPMENT = 0.8;
FRACTION_TESTING = 0.2;
FRACTION_TRAINING = 0.8;
FRACTION_VALIDATION = 0.2;
FUNCTIONS_NORMALIZATION = ["zscore", "norm", "range"];
N_PROJECTION_FEATURES = 25;
N_TOP_DISCRIMINANT_KW_RANKED_FEATURES = 15;
N_TOP_DISCRIMINANT_RF_RANKED_FEATURES = 8;

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
    target_num(ismember(csv_data.label, target_labels{i})) = i;
end

features = table2array(csv_data(:, 2:end - 1));
features = to_data_struct(features, target_num);

% PCA

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
    %save_img(file_path);

    % Kruskal Wallis

    [features_h, features_idx] = rank_kruskal_wallis(features_);
    top_features = features_h(1:N_TOP_DISCRIMINANT_KW_RANKED_FEATURES);
    top_features_idx = features_idx(1:N_TOP_DISCRIMINANT_KW_RANKED_FEATURES);
    
    fprintf("Applying Kruskal-Wallis test...\n");
    fprintf("-------Top %d KW ranked features-------\n", N_TOP_DISCRIMINANT_KW_RANKED_FEATURES);

    for j=(1:N_TOP_DISCRIMINANT_KW_RANKED_FEATURES)
        fprintf("%d - %s - H = %.3f\n", j, col_names_{features_idx(j)}, features_h(j));
    end
    
    features_.X = features_.X(top_features_idx, :); % Update the features to the ones most relevant
    col_names_ = col_names_(top_features_idx);
    features_.dim = size(features_.X, 1);
    %test_norm(features_, col_names_);
    
    figure;
    scatter((1:size(features_h, 2)), features_h);
    file_path = PATH_PLOT_IMAGES + "scatter_h_kw" + EXTENSION_IMG;
    save_img(file_path);

    figure;
    ppatterns(features_);
    file_path = PATH_PLOT_IMAGES + "patterns_pca_kw" + EXTENSION_IMG;
    %save_img(file_path);

    
    % End of Kruscal Wallis
    
    % Correlation study
    
    fprintf("Calculating %d top KW features correlation matrix...\n", N_TOP_DISCRIMINANT_KW_RANKED_FEATURES);
    corr_matrix = corrcoef(features_.X'); % Only 20 best
    figure;
    heatmap(col_names_, col_names_, corr_matrix);
    file_path = PATH_PLOT_IMAGES + "corr_matrix_kw" + EXTENSION_IMG;
    %save_img(file_path);
    
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
        fprintf("%d - %s - I = %.3f\n", j, col_names_{idx(j)}, importances(j));
    end
    
    file_path = PATH_PLOT_IMAGES + "importance_rf" + EXTENSION_IMG;
    %save_img(file_path);
    
    % End of RandomForest
end

% End of PCA

%[train_data, val_data, test_data] = divide_data(features, FRACTION_TRAINING, ...
%    FRACTION_VALIDATION,FRACTION_TESTING); % Divide the data for training, validation and testing

% Minimum distance classifier

data_mdc = features_; % Eu pensei que isto era a data struct de features reduzidas, mas tá 10x3? não é muito pouco?

for i=(1:size(data_mdc.y))
    
    choice_class = i;
    
    % First, set all entries in the y field to 1
    data_mdc.y = ones(size(data_mdc.y));
    
    % Then, set the entries corresponding to the choice class to 0
    data_mdc.y(data_mdc.y == choice_class) = 0;
    
    % Run the Minimum Distance Classifier
    [accuracy, errors] = mdc_euclidean(data_mdc);

    fprintf("Accuracy for class %d: %f\n", i, accuracy);
    
    % Reset labels
    data_mdc.y = features_.y;

end

% End of Minimum distance classifier



