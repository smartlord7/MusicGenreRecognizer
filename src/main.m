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
N_TOP_DISCRIMINANT_RANKED_FEATURES = 30;
N_PROJECTION_FEATURES = 10;

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

% Kruscal Wallis
features = table2array(csv_data(:, 2:end - 1));
features = to_data_struct(features, target_num);
[features_h, features_idx] = rank_kruskal_wallis(features);

top_n_features_h = features_h(1:N_TOP_DISCRIMINANT_RANKED_FEATURES);
top_n_features_idx = features_idx(1:N_TOP_DISCRIMINANT_RANKED_FEATURES);

fprintf("Applying Kruskal-Wallis test...\n");
fprintf("-------Top %d KW ranked features-------\n", N_TOP_DISCRIMINANT_RANKED_FEATURES)
for i=(1:N_TOP_DISCRIMINANT_RANKED_FEATURES)
    fprintf("%d - %s - H = %.3f\n", i, col_names{features_idx(i)}, features_h(i));
end

features.X = features.X(top_n_features_idx, :); % Update the features to the ones most relevant
col_names = col_names(top_n_features_idx);
%test_norm(features, col_names);

figure;
scatter((1:features.dim), features_h);

% End of Kruscal Wallis

% Correlation study

fprintf("Calculating %d top KW features correlation matrix...\n", N_TOP_DISCRIMINANT_RANKED_FEATURES);
corr_matrix = corrcoef(features.X'); % Only 20 best
figure;
heatmap(col_names, col_names, corr_matrix);
file_path = PATH_PLOT_IMAGES + "corr_matrix_kw" + EXTENSION_IMG;
save_img(file_path);

% End of correlation study

% PCA

for i=(1:size(FUNCTIONS_NORMALIZATION, 2))
    norm_function = FUNCTIONS_NORMALIZATION(i);
    fprintf("Normalizing using '%s'...\n", norm_function);

    features.X = normalize(features.X, norm_function);
    x = features.X;
    fprintf("Applying PCA for %d dimensions...\n", N_PROJECTION_FEATURES);
    [proj, eigenValues, eigenVectors, individual_variance] = project_pca(features, N_PROJECTION_FEATURES);
    
    figure;
    plot(eigenValues, 'o-.');
    xlabel('Principal component');
    ylabel('Eigen value');
    file_path = PATH_PLOT_IMAGES + "pca_eigen_values" + norm_function + EXTENSION_IMG;
    save_img(file_path);
    
    figure;
    plot(individual_variance, 'o-');
    xlabel('Principal component');
    ylabel('% of variance');
    file_path = PATH_PLOT_IMAGES + "pca_variance" + norm_function + EXTENSION_IMG;
    save_img(file_path);

    % Correlation study
    
    fprintf("Calculating %d PCA features correlation matrix...\n", N_PROJECTION_FEATURES);
    corr_matrix = corrcoef(proj.X'); % Only the ones most important in PCA analysis
    
    figure;
    heatmap((1:N_PROJECTION_FEATURES), (1:N_PROJECTION_FEATURES), corr_matrix);
    file_path = PATH_PLOT_IMAGES + "corr_matrix_pca" + norm_function + EXTENSION_IMG;
    save_img(file_path);

    figure;
    ppatterns(proj);

    % End of correlation study
end

% End of PCA

%[train_data, val_data, test_data] = divide_data(features, FRACTION_TRAINING, ...
%    FRACTION_VALIDATION,FRACTION_TESTING); % Divide the data for training, validation and testing



