% Clear figures and workspace variables
close all
clear

% Make this program reproducible
rng(0)

[col_names, features, target_labels] = import_dataset(Const.PATH_FEATURES, "GTZAN");

results_table = cell(size(Const.FUNCTIONS_NORMALIZATION, 2) * ...
    size(Const.FUNCTIONS_DISTANCES, 2) * ...
    size(target_labels, 2), 8);

count = 1;
metadata = struct;
metadata.n_top_features_kw = Const.N_TOP_DISCRIMINANT_KW_RANKED_FEATURES;
metadata.n_top_features_rf = Const.N_TOP_DISCRIMINANT_RF_RANKED_FEATURES;
metadata.dim_pca = Const.N_PROJECTION_FEATURES;
metadata.base_path = Const.PATH_PLOT_IMAGES;
metadata.ext = Const.EXTENSION_IMG;
metdata.plot = true;

for i=(1:size(Const.FUNCTIONS_NORMALIZATION, 2))
    norm_function = Const.FUNCTIONS_NORMALIZATION(i);
    metadata.norm_function = norm_function;
    features_ = features;

    fprintf("Normalizing using '%s'...\n", norm_function);
    features_.X = normalize(features_.X, norm_function);

    features_ = rank_kruskal_wallis(features_, metadata);
    
    features_ = rank_random_forest(features_, metadata);

    features_ = project_pca(features_, metadata);
    

    % Minimum distance classifier

    data_mdc = features_; 
    [train_data, val_data, test_data] = divide_data(features_, Const.FRACTION_TRAINING, ...
    Const.FRACTION_VALIDATION, Const.FRACTION_TESTING); % Divide the data for training, validation and testing

    % TESTING  HERE ------------------------------------
    for j=(1:size(target_labels))
        train_data_ = train_data;
        val_data_ = val_data;

        choice_classt = j - 1;
        %choice_classt = 0;
        traindata = train_data;
        valdata = val_data;
        train_data_bin = to_bin_classification(train_data, choice_classt);
        val_data_bin = to_bin_classification(val_data, choice_classt);
    
        [predicted_train, predicted_test] = min_dist_classifier(train_data_bin, val_data_bin, "mahalanobis" , "Binary");
        %[predicted_train, predicted_test] = fisher_LDA(train_data_bin, val_data_bin);
        
        %[mse, accuracy, specificity, sensitivity, f_measure, auc] = eval_classifier(val_data_bin.y, predicted_test, LABELS_BINARY, 'a');
        %[predicted_train, predicted_test] = bayes_classifier(traindata, valdata, "Multi-Class");
        %[predicted_train, predicted_test] = KNN_classifier(traindata, valdata);
        %[predicted_train, predicted_test] = SVM_classifier(train_data_bin, val_data_bin, 'Binary');
    
    end
    % FINISH TESTING  HERE -----------------------------
 
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

        for k=(1:size(Const.FUNCTIONS_DISTANCES, 2))
            dist_func = Const.FUNCTIONS_DISTANCES(k);

            % Run the Minimum Distance Classifier
            [predicted_train, predicted_test] = min_dist_classifier(train_data_, val_data_, dist_func , "Binary");
            file_path = Const.PATH_PLOT_IMAGES + "cm_" + genre + "_" + norm_function + "_" + dist_func + Const.EXTENSION_IMG;
            [mse, accuracy, specificity, sensitivity, f_measure, auc] = eval_classifier(val_data_.y, predicted_test, Const.LABELS_BINARY, file_path);
            fprintf("MSE: %.3f\n" + ...
                "Accuracy: %.3f\n" + ...
                "Specificity: %.3f\n" + ...
                "Sensitivity: %.3f\n" + ...
                "F-measure: %.3f\n"+ ...
                "Auc: %.3f\n", mse, accuracy, specificity, sensitivity, f_measure, auc);
            
             % store the results in a cell array
                results_table{count, 1} = Const.FUNCTIONS_NORMALIZATION(i);
                results_table{count, 2} = Const.FUNCTIONS_DISTANCES(k);
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

