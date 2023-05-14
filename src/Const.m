classdef Const
    %CONST Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(Constant)
        % Constants
        DIRECTORY_DATA = "../data/";
        EXTENSION_FEATURES = ".csv";
        EXTENSION_IMG = ".png";
        PATH_PLOT_IMAGES = "../img/plots/";
        PATH_FEATURES = Const.DIRECTORY_DATA + "gtzan" + Const.EXTENSION_FEATURES;
        FRACTION_DEVELOPMENT = 0.8;
        FRACTION_TESTING = 0.2;
        FRACTION_TRAINING = Const.FRACTION_DEVELOPMENT * 0.8;
        FRACTION_VALIDATION = Const.FRACTION_DEVELOPMENT * 0.2;
        FUNCTIONS_NORMALIZATION = ["zscore", "norm", "range"];
        FUNCTIONS_DISTANCES = ["euclidean", "cityblock", "minkowski", "chebychev", "mahalanobis"];
        LABELS_BINARY = {'Negative', 'Positive'};
        N_PROJECTION_FEATURES = 8;
        N_TOP_DISCRIMINANT_KW_RANKED_FEATURES = 30;
        N_TOP_DISCRIMINANT_RF_RANKED_FEATURES = 10;
        PLOT = false;
        N_PARTITIONS = 10;
    end 
end

