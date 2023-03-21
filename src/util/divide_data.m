function [train_data, val_data, test_data] = divide_data(data, train_fraction, val_fraction, test_fraction)
    x = data.X;
    y = data.y;
    idx = randperm(data.num_data);
    x = x(:, idx);
    y = y(:, idx);
    l = data.num_data;
    [train_ind, val_ind, test_ind] = divideblock(l, train_fraction, val_fraction, test_fraction);
    train_x = x(:, train_ind);
    val_x = x(:, val_ind);
    test_x = x(:, test_ind);

    train_y = y(:, train_ind);
    val_y = y(:, val_ind);
    test_y = y(:, test_ind);

    train_data = to_data_struct(train_x', train_y');
    val_data = to_data_struct(val_x', val_y');
    test_data = to_data_struct(test_x', test_y');
end

