function [data] = normZscore(data)
    x = data.X;
    m = mean(x, 2);
    s = std(x, 1, 2);
    s(s == 0) = 1;

    data.X = (x - m) ./ s;    
end