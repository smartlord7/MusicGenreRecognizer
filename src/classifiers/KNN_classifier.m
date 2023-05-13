function [ypred2_train, ypred2_val] = KNN_classifier(train_data, val_data)
%KNN_CLASSIFIER Summary of this function goes here

% Choose a range of values for K
k_range = 10;

% Repeat the procedure for several different random permutations of the data
n_permutations = 10;
models = cell(n_permutations, k_range);
err = zeros(n_permutations, k_range);

for j = 1:n_permutations
    % Permute the data
    permuted_idx_trn = randperm(train_data.num_data);
    train_data.X = train_data.X(:, permuted_idx_trn);
    train_data.y = train_data.y(permuted_idx_trn);

    permuted_idx_tst = randperm(val_data.num_data);
    val_data.X = val_data.X(:, permuted_idx_tst);
    val_data.y = val_data.y(permuted_idx_tst);
    
    % Train and evaluate KNN classifiers with different values of K
    for i = 1:k_range
        clear model
        model = knnrule(train_data, i);
        models{j,i} = model;

        ypred = knnclass(val_data.X, model);
        err(i,j)=cerror(ypred,val_data.y)*100;
        plot(err(j,1:i))
        drawnow

    end
end

merr=mean(err);
serr=std(err);

ix=find(merr==min(merr));
ix=ix(1);
ix_min_err=find(err(:,ix)==min(err(:,ix)));

best=models{ix_min_err, ix};
%end
ypred2_train = predict(best, train_data.X');
ypred2_val = predict(best, val_data.X');
err2=cerror(ypred2', val_data.y)*100;


end

