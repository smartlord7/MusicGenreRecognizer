function [ypred, error, accuracy] = fisher_LDA(train_data, val_data)
%FISHER_LDA Summary of this function goes here

% Change target labels from 0 and 1 to 1 and 2, respectively
% Esta função quer targets 1 e 2..
target_labels = [0 1];
new_labels = [1 2];
[~, target_train] = ismember(train_data.y, target_labels);
[~, target_val] = ismember(val_data.y, target_labels);
train_data.y = new_labels(target_train);
val_data.y = new_labels(target_val);

% Use the fld function to obtain the discriminant function and bias term
model = fld(train_data);

% Test
ypred = linclass(val_data.X, model);
figure; ppatterns(train_data); pline(model);

% Isto foi mais pa testar imediatamente sem o ciclo
error = cerror(ypred, val_data.y);
C = confusionmat(val_data.y, ypred);
accuracy = sum(diag(C)) / sum(C(:));

% Display the confusion matrix
disp('Confusion matrix:');
disp(C);

end

