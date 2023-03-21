function [importances] = rank_rf_importance(data)
    x = data.X';
    y = data.y;
    rf = TreeBagger(50, x, y, ...
        PredictorSelection="interaction-curvature", ...
        OOBPredictorImportance="on", ...
        Surrogate="on", ...
        OOBPredictorImportance="on");

    importances = rf.OOBPermutedPredictorDeltaError;
    
%     figure;
%     bar(importances);
%     title("Interaction-Curvature Test")
%     ylabel("Predictor Importance Estimates")
%     xlabel("Predictors")
%     h = gca;
%     h.XTickLabel = rf.PredictorNames;
%     h.XTickLabelRotation = 45;
%     h.TickLabelInterpreter = "none";
end

