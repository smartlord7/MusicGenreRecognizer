% Clear figures and workspace variables
close all
clear
% Make this program reproducible
rng(0)
[col_names, features, target_labels] = import_dataset(Const.PATH_FEATURES, "GTZAN");
results_table = cell(size(Const.FUNCTIONS_NORMALIZATION, 2) * ...
size(target_labels, 2), 8);
metadata = struct;
metadata.n_top_features_kw = Const.N_TOP_DISCRIMINANT_KW_RANKED_FEATURES;
metadata.n_top_features_rf = Const.N_TOP_DISCRIMINANT_RF_RANKED_FEATURES;
metadata.dim_pca = Const.N_PROJECTION_FEATURES;
metadata.base_path = Const.PATH_PLOT_IMAGES;
metadata.ext = Const.EXTENSION_IMG;
metadata.plot = true;

count = 1;

for i=(1:size(Const.FUNCTIONS_NORMALIZATION, 2))
    norm_function = Const.FUNCTIONS_NORMALIZATION(i);
    metadata.norm_function = norm_function;
    features_ = features;
    
    fprintf("Normalizing using '%s'...\n", norm_function);
    features_.X = normalize(features_.X, norm_function);
    
    features_ = rank_kruskal_wallis(features_, metadata);
    
    features_ = rank_random_forest(features_, metadata);
    
    features_ = project_pca(features_, metadata);

    for j=(1:size(target_labels))
        genre = target_labels{j};
        
        
        % Initialize arrays to store the performance metrics for each partition
        mse_array = zeros(Const.N_PARTITIONS, 1);
        accuracy_array = zeros(Const.N_PARTITIONS, 1);
        
        % Classify multiple random partitions of the data
        cv = cvpartition(features_.y, 'KFold', Const.N_PARTITIONS);
        for k = 1:Const.N_PARTITIONS
            train_data_partitionX = features_.X(:, cv.training(k));
            train_data_partitionY = features_.y(:, cv.training(k));
            val_data_partitionX = features_.X(:, cv.test(k));
            val_data_partitionY = features_.y(:, cv.test(k));
            sT = struct;
            sT.X = train_data_partitionX;
            sT.y = train_data_partitionY;
            sV = struct;
            sV.X = val_data_partitionX;
            sV.y = val_data_partitionY;
            sT.dim = size(train_data_partitionX, 1);
            sV.dim = size(val_data_partitionX, 1);
            %[predicted_train, predicted_test] = random_forest_classifier(sT, sV);
            %[predicted_train, predicted_test] = min_dist_classifier(sT, sV, "mahalanobis", "Binary");
            %[predicted_train, predicted_test] = fisher_LDA_classifier(sT, sV);
            %[predicted_train, predicted_test] = SVM_classifier(sT, sV, "Multiclass");
            %[predicted_train, predicted_test] = bayes_classifier(sT, sV, "Multiclass");
            [predicted_train, predicted_test] = KNN_classifier(sT, sV);
            [mse, accuracy, specificity, sensitivity, f_measure, auc] = eval_classifier(val_data_partitionY', predicted_test', Const.LABELS_BINARY, "knn" + genre + "_" + norm_function + string(k));
    
            % Store the performance metrics for the current partition
            mse_array(k) = mse;
            accuracy_array(k) = accuracy;
            specificity_array(k) = specificity;
            sensitivity_array(k) = sensitivity;
            f_measure_array(k) = f_measure;
            auc_array(k) = auc;
        end
    
        % Calculate the mean and standard deviation of the performance metrics across all partitions
        mse_mean = mean(mse_array);
        mse_std = std(mse_array);
        accuracy_mean = mean(accuracy_array);
        accuracy_std = std(accuracy_array);
        specificity_mean = mean(specificity_array);
        specificity_std = std(specificity_array);
        sensitivity_mean = mean(sensitivity_array);
        sensitivity_std = std(sensitivity_array);
        f_measure_mean = mean(f_measure_array);
        f_measure_std = std(f_measure_array);
        auc_mean = mean(auc_array);
        auc_std = std(auc_array);
    
        fprintf("MSE: %.3f +/- %.3f\n" + ...
                "Accuracy: %.3f +/- %.3f\n" + ...
                "Specificity: %.3f +/- %.3f\n" + ...
                "Sensitivity: %.3f +/- %.3f\n" + ...
                "F-measure: %.3f +/- %.3f\n" + ...
                "Auc: %.3f +/- %.3f\n", mse_mean, mse_std, accuracy_mean, accuracy_std, specificity_mean, specificity_std, sensitivity_mean, sensitivity_std, f_measure_mean, f_measure_std, auc_mean, auc_std);
    
         % store the results in a cell array
            results_table{count, 1} = Const.FUNCTIONS_NORMALIZATION(i);
            results_table{count, 2} = string(genre);
            results_table{count, 3} = mse_mean;
            results_table{count, 4} = accuracy_mean;
            results_table{count, 5} = specificity_mean;
            results_table{count, 6} = sensitivity_mean;
            results_table{count, 7} = f_measure_mean;
            results_table{count, 8} = auc_mean;

            count = count + 1;     
    
    end
end
% convert the cell array to a table
header = {'Normalization function', ...
'Class', 'MSE', 'Accuracy', 'Specificity', 'Sensitivity', 'F-measure', 'AUC'};
results_table = cell2table(results_table, 'VariableNames', header);
% write the table to an Excel file
writetable(results_table, 'knn_multi.xlsx');
% End of Minimum distance classifier