function [ypred_train, ypred_val] = SVM_classifier(train_data, val_data, classif_type)
%SVM_CLASSIFIER Summary of this function goes here

if (classif_type == "Binary")

    c_pot=[-20:6];
    C=2.^c_pot;
    
    % Repeat the procedure for several different random permutations of the data
    n_permutations = 1;
    models = cell(n_permutations, numel(C));
    err = zeros(n_permutations, numel(C));

    for n = 1:n_permutations
        % Permute the data
        permuted_idx_trn = randperm(train_data.num_data);
        train_data.X = train_data.X(:, permuted_idx_trn);
        train_data.y = train_data.y(permuted_idx_trn);
    
        permuted_idx_tst = randperm(val_data.num_data);
        val_data.X = val_data.X(:, permuted_idx_tst);
        val_data.y = val_data.y(permuted_idx_tst);
        
        % Train and evaluate SVM classifiers with different values of C
        for co=1:numel(C)
            disp('Train');
            model = fitcsvm(train_data.X', train_data.y', 'KernelFunction','linear', 'BoxConstraint',C(co),'Solver','SMO');
            [ypred]= predict(model,train_data.X');
            err(n,co)=cerror(ypred', train_data.y)*100;
            models{n,co}=model;
        end
    end
    
    if n_permutations > 1
        merr=mean(err);
        serr=std(err);
    else
        merr=err;
        serr=zeros(1,numel(err));
    end
    
    ix=find(merr==min(merr));
    ix=ix(1);
    ix_min_err=find(err(:,ix)==min(err(:,ix)));
    
    best=models{ix_min_err, ix};
    %end
    ypred_train = predict(best, train_data.X');
    ypred_val = predict(best, val_data.X');
    %err2=cerror(ypred2', val_data.y)*100;

else % If Multi-Class

    n_classes = length(unique(train_data.y));
    models = cell(n_classes, 1);

    for i = 1:n_classes
        % Create a binary label vector for this class vs. all others
        binary_labels = double(train_data.y == i - 1);

        % Train SVM model
        model = fitcsvm(train_data.X', binary_labels, 'KernelFunction','rbf', 'BoxConstraint',1);

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

