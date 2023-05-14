function [ypred_train, ypred_val] = fisher_LDA_classifier(train_data, val_data)
    %FISHER_LDA Summary of this function goes here
    
    % Change target labels from 0 and 1 to 1 and 2, respectively
    train_data.y = train_data.y + 1;
    val_data.y =  val_data.y + 1;
    
    % Use the fld function to obtain the discriminant function and bias term
    model = fld(train_data);
    
    % Test
    ypred_train = linclass(train_data.X, model) - 1;
    ypred_val = linclass(val_data.X, model) - 1;

end

