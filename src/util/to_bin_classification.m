function [data] = to_bin_classification(data, choice_class)
    y = data.y;

    class_idx = find(y == choice_class);
    non_class_idx = find(y ~= choice_class);
    
    data.y(class_idx) = 1;
    data.y(non_class_idx) = 0;
end

