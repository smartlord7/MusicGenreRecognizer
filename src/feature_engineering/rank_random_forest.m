function [features] = rank_random_forest(features, metadata)
    n_top_features = metadata.n_top_features_rf;
    base_path = metadata.base_path;
    ext = metadata.ext;

    [importances, rf] = rank_random_forest_(features);
    [importances, idx] = sort(importances, "descend");
    idx = idx(1:n_top_features);
    features.X = features.X(idx, :);
    features.dim = size(features.X, 1);

    fprintf("Applying Random Forest test...\n");
    fprintf("-------Top %d RF ranked features by importance-------\n", n_top_features);

    for j=(1:n_top_features)
        fprintf("%d - %d - I = %.3f\n", j, idx(j), importances(j));
    end
    
    if Const.PLOT == true
        figure;
        bar(importances);
        title("Interaction-Curvature Test")
        ylabel("Predictor Importance Estimates")
        xlabel("Predictors")
        h = gca;
        h.XTickLabel = rf.PredictorNames;
        h.XTickLabelRotation = 45;
        h.TickLabelInterpreter = "none";
        file_path = base_path + "importance_rf" + ext;
        save_img(file_path);
    end  
end
