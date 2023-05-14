function [ypred_train, ypred_val] = SVM_classifier(train_data, val_data, classif_type)
%SVM_CLASSIFIER performs classification using a support vector machine (SVM) classifier.
if (classif_type == "Binary")
    % If binary classification is selected:
    % Define a range of values for C and gamma
    c_pot=[-5:5];
    C=2.^c_pot;
    g_pot=[-6:-1];
    G=2.^g_pot;
    
    % Initialize models and errors matrices
    models = cell(numel(C), numel(G));
    err = zeros(numel(C), numel(G));
    
    % Train and evaluate SVM classifiers with different values of C and gamma
    for co=1:numel(C)
        for go=1:numel(G)
            % Train an SVM classifier with a polynomial kernel using the
            % training data and the current values of C and gamma
            fprintf("C: %d, G: %d\n", C(co), G(go))
            model = fitcsvm(train_data.X', train_data.y', 'KernelFunction','polynomial', 'BoxConstraint',C(co), 'KernelScale', sqrt(1/(2*G(go))), 'Solver','SMO');
            
            % Compute the predicted labels for the validation set and
            % calculate the classification error
            ypred = predict(model,val_data.X');
            err(co, go) = cerror(ypred', val_data.y) * 100;
            
            % Store the model and its error
            models{co, go} = model;
        end
    end
    
    % Find the model with the lowest error on the validation set
    [~, idx] = min(err(:));
    [idx_min_err, idx_max_err] = ind2sub(size(err), idx);
    best_model = models{idx_min_err, idx_max_err};
    
    % Compute the predicted labels for the training and validation sets
    ypred_train = predict(best_model, train_data.X');
    ypred_val = predict(best_model, val_data.X');
else % If Multi-Class
    % If multi-class classification is selected:
    
    
    % Determine the number of classes
    n_classes = length(unique(train_data.y));
    
    % Initialize a cell array to store the SVM models
    models = cell(n_classes, 1);
    
    % Train one SVM model per class
    for i = 1:n_classes
        % Create a binary label vector for this class vs. all others
        binary_labels = double(train_data.y == i - 1);
    
        % Train an SVM model with a linear kernel using the training data
        model = fitcsvm(train_data.X', binary_labels, 'KernelFunction','linear', 'BoxConstraint',1);
    
        % Store the model
        models{i} = model;
    end
    
    % Compute the predicted labels for the training and validation sets
    scores_train = zeros(size(train_data.X,2), n_classes);
    scores_test = zeros(size(val_data.X,2), n_classes);
    for i = 1:n_classes
        % Compute the scores for the positive class (i.e., the current class)
        [~, score] = predict(models{i}, train_data.X');
        scores_train(:, i) = score(:, 2);
    
        [~, score] = predict(models{i}, val_data.X');
        scores_test(:, i) = score(:, 2);
    end

    % Determine the predicted labels by selecting the class with the
    % highest score

    [~, ypred_train] = max(scores_train, [], 2);
    ypred_train = ypred_train';
    ypred_train = ypred_train - 1;

    [~, ypred_val] = max(scores_test, [], 2);
    ypred_val = ypred_val';
    ypred_val = ypred_val - 1;
end % End Multi-Class