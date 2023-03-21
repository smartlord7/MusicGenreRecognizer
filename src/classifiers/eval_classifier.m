function [mse, accuracy, specificity, sensitivity, f_measure] = eval_classifier(y, y_predicted, plot_path)
    n_samples = size(y, 2);

    % Compute mean squared error
    mse = (sum(y - y_predicted) .^ 2) / n_samples;
    
    % Compute confusion matrix
    conf_mat = confusionmat(y, y_predicted);
    
    % Create confusion chart
    labels = {'Negative', 'Positive'};
    f = figure(1);
    plotconfusion(y, y_predicted)
    fh = gcf;
    ax = gca;
    ax.FontSize = 8;
    set(findobj(ax,'type','test'),'fontsize',3);
    ah = fh.Children(2);
    ah.XLabel.String = 'Actual';
    ah.YLabel.String = 'Predicted';
    ax.XTickLabel = labels;
    ax.YTickLabel = labels;
    title("");
    hold off;
    
    save_img(plot_path);

    % Compute accuracy
    accuracy = sum(diag(conf_mat))/sum(conf_mat(:));
    
    % Compute specificity
    specificity = conf_mat(2,2) / (conf_mat(2,2) + conf_mat(2,1));
    
    % Compute sensitivity (recall)
    sensitivity = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(1,2));
    
    % Compute F-measure (F1-score)
    precision = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(2,1));
    f_measure = 2 * (precision * sensitivity) / (precision + sensitivity);
end

