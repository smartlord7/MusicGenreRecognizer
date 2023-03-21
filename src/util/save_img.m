function [] = save_img(path)
    set(gcf, 'Position', get(0, 'Screensize'));
    saveas(gcf, path);
    close all;
end

