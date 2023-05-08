function [y_predicted, error] = bayes_classifier(train_data, val_data, classif_type)
%MDC_EUCLIDEAN Summary of this function goes here

if (classif_type == "Binary")
    n_classes = 2;
else 
    n_classes = 10;
end

% Initialize the model struct
model.Pclass = cell(1, n_classes);
model.Prior = zeros(1, n_classes);
model.fun = 'bayescls';

% Loop over each class
for i = 1:n_classes
    % Extract the training data for the current class
    idx = find(train_data.y == (i-1));
    trn_data.X = train_data.X(:, idx);
    trn_data.y = ones(length(idx), 1);
    trn_data.dim = size(train_data.X, 2);
    
    % Estimate the likelihood for the current class
    model.Pclass{i} = mlcgmm(trn_data);
    
    % Estimate the prior probability for the current class
    model.Prior(i) = length(idx) / length(train_data.y);
end

% Make predictions using the trained model
y_predicted = bayescls(train_data.X, model);
y_predicted = y_predicted - 1; % Predicts are 1 and 2.. 
error = cerror(y_predicted, train_data.y);

end

