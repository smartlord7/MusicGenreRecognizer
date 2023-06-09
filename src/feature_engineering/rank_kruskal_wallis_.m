function [features_h, features_idx] = rank_kruskal_wallis_(data)
    dim = data.dim;
    hAll = zeros(1, dim);
    for i=(1:dim)
        [~, table, ~] = kruskalwallis(data.X(i, :), data.y, 'off');
        h = table{2, 5};
        if isnan(h)
            h = 0;
        end
        hAll(i) = h;
    end
    
    [features_h, features_idx] = sort(hAll, 'descend');
end
