function [] = test_norm(data, col_names)
    for i=(1:data.dim)
        figure;
        feature = string(col_names(i));
        x = data.X(i, :)';
        histfit(x);
        title(feature);
        h = kstest(x);
        fprintf("%s normal? %d\n", feature, h);
        figure;
        cdfplot(x)
        hold on;
        x_values = linspace(min(x), max(x));
        plot(x_values, normcdf(x_values, 0, 1),'r-')
        title(feature);
        legend('Empirical CDF', 'Standard Normal CDF', 'Location','best');
    end
end

