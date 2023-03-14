function [train_data, val_data, test_data] = divide_data(data, train_fraction, val_fraction, test_fraction)
    l = size(data, 1);
    [train_ind, val_ind, test_ind] = dividerand(l, train_fraction, val_fraction, test_fraction);
    train_data = data(train_ind, :);
    val_data = data(val_ind, :);
    test_data = data(test_ind, :);
end

