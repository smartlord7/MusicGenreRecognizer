function [mse, accuracy, specificity, sensitivity, f_measure, auc] = eval_classifier(y, y_predicted, labels, plot_path)
    n_samples = size(y, 2);

    % Compute mean squared error
    mse = (sum(y - y_predicted) .^ 2) / n_samples;
    
    % Compute confusion matrix
    conf_mat = confusionmat(y, y_predicted);
    
    % Create confusion chart
    if Const.PLOT == true
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
    end

    % Compute accuracy
    accuracy = sum(diag(conf_mat))/sum(conf_mat(:));

    if length(unique(y)) == 2
    
        % Compute specificity
        specificity = conf_mat(2,2) / (conf_mat(2,2) + conf_mat(2,1));
        
        % Compute sensitivity (recall)
        sensitivity = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(1,2));
        
        % Compute F-measure (F1-score)
        precision = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(2,1));
        f_measure = 2 * (precision * sensitivity) / (precision + sensitivity);

        % Compute ROC curve and AUC
        [fpr, tpr] = perfcurve(y, y_predicted, '1');

        % Compute AUC
        auc = trapz(fpr, tpr);
        
        % Plot ROC curve
        if Const.PLOT == true
            f = figure;
            plot(fpr, tpr)
            xlabel('False Positive Rate')
            ylabel('True Positive Rate')
            title('ROC Curve')
            grid on
            save_img(plot_path);
        end
    else
        specificity = nan(1);
        sensitivity = nan(1);
        precision = nan(1);
        f_measure = nan(1);
        auc = nan(1);
    end
end

