function [ypred_train, ypred_val] = KNN_classifier(train_data, val_data)
%KNN_CLASSIFIER Summary of this function goes here

% Choose a range of values for K
k_range = 25;

% Initialize models and error matrix
models = cell(1, k_range);
err = zeros(1, k_range);


% Train and evaluate KNN classifiers with different values of K
for i = 1:k_range
    clear model
    model = knnrule(train_data, i);
    models{i} = model;

    ypred = knnclass(val_data.X, model);
    err(i) = cerror(ypred, val_data.y) * 100;
    plot(err(1:i))
    drawnow
end


ix=find(err==min(err));
ix=ix(1);
best = models{ix};

%end
ypred_train = knnclass(train_data.X, best);
ypred_val = knnclass(val_data.X, best);

end

