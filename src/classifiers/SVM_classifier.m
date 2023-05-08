function [outputArg1,outputArg2] = SVM_classifier(train_data, val_data)
%SVM_CLASSIFIER Summary of this function goes here

c_pot=[-20:6];
C=2.^c_pot;

% Repeat the procedure for several different random permutations of the data
n_permutations = 10;
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
        disp(sprintf('=========\nrun=%d\nCost=%f\n===========', n,C(co)));
        disp('Train');
        model = fitcsvm(train_data.X', train_data.y', 'KernelFunction','linear', 'BoxConstraint',C(co),'Solver','SMO');
        [ypred]= predict(model,val_data.X');
        err(n,co)=cerror(ypred', val_data.y)*100;
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
[ypred2,dfce] = predict(best, val_data.X');
err2=cerror(ypred2', val_data.y)*100;

end

