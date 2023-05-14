function [ypred_train, ypred_val] = SVM_classifier(train_data, val_data, classif_type)
%SVM_CLASSIFIER Summary of this function goes here

if (classif_type == "Binary")

    c_pot=[-10:12];
    C=2.^c_pot;
    g_pot=[-20:0];
    G=2.^g_pot;
    
    
    models = cell(numel(C), numel(G));
    err = zeros(numel(C), numel(G));

        
    % Train and evaluate SVM classifiers with different values of C
 
    for co=1:numel(C)
        for go=1:numel(G)
            disp('Train');
            model = fitcsvm(train_data.X', train_data.y', 'KernelFunction','rbf', 'BoxConstraint',C(co), 'kernelScale', sqrt(1/(2*G(go))), 'Solver','SMO');
            [ypred]= predict(model,val_data.X');
            err(co, go)=cerror(ypred', val_data.y)*100;
            models{co, go}=model;
        end
    end
    
    
    
    merr=err;
    serr=zeros(1,numel(err));
    
    
    ix=find(merr==min(merr));
    ix=ix(1);
    ix_min_err=find(err(:,ix)==min(err(:,ix)));
    
    best=models{ix_min_err, ix};
    %end
    ypred_train = predict(best, train_data.X');
    ypred_val = predict(best, val_data.X');
    err2=cerror(ypred_val', val_data.y)*100;
    err2=cerror(ypred2', val_data.y)*100;

else % If Multi-Class

    n_classes = length(unique(train_data.y));
    models = cell(n_classes, 1);

    for i = 1:n_classes
        % Create a binary label vector for this class vs. all others
        binary_labels = double(train_data.y == i - 1);

        % Train SVM model
        model = fitcsvm(train_data.X', binary_labels, 'KernelFunction','linear', 'BoxConstraint',1);

        % Store model
        models{i} = model;
    end

    % Compute the predicted labels for the validation set
    disp('Predict');
    scores_train = zeros(size(train_data.X,2), n_classes);
    scores_test = zeros(size(val_data.X,2), n_classes);
    for i = 1:n_classes
        [~, score] = predict(models{i}, train_data.X');
        scores_train(:, i) = score(:, 2); % use scores for the positive class

        [~, score] = predict(models{i}, val_data.X');
        scores_test(:, i) = score(:, 2); % use scores for the positive class
    end
    [~, ypred_train] = max(scores_train, [], 2);
    ypred_train = ypred_train';
    ypred_train = ypred_train - 1;

    [~, ypred_val] = max(scores_test, [], 2);
    ypred_val = ypred_val';
    ypred_val = ypred_val - 1;

    % Compute the classification error rate
    err_train=cerror(ypred_train, train_data.y)*100;
    % Compute the classification error rate
    err_val=cerror(ypred_val, val_data.y)*100;

end % End Multi-Class

end

