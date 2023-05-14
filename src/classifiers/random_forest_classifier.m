function [y_predicted_train, y_predicted_val] = random_forest_classifier(train_data, val_data)
%RANDOM_FOREST_CLASSIFIER performs classification using a random forest classifier.
% Set the number of trees in the forest
n_trees = 100;
% Train the random forest classifier
model = TreeBagger(n_trees, train_data.X', train_data.y', 'Method', 'classification');
% Make predictions using the trained model
y_predicted_train = predict(model, train_data.X');
y_predicted_val = predict(model, val_data.X');
% Convert the predicted labels to numeric format
y_predicted_train = str2double(y_predicted_train);
y_predicted_val = str2double(y_predicted_val);
% Calculate the classification error for the training and validation sets
end