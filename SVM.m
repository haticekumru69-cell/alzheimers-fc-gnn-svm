%% =========================================================
%  STEP 3: SVM-BASED FEATURE SELECTION and EVALUATION
%% =========================================================

clear; clc; close all;

%% =========================================================
% 1. DATA LOADING
%% =========================================================

data_folder = fullfile(pwd, 'ALZHEIMER_SVM_HON_SPEARMAN');
file_path   = fullfile(data_folder, 'Alzheimer_HonSpearman_ML.mat');

if ~exist(file_path, 'file')
    error(' Data file not found!');
end

load(file_path);   % -> loads SVM_Data

X = SVM_Veri.X;   % (N x 400)
y = SVM_Veri.y;   % (N x 1)

fprintf(' Data loaded: %d samples, %d features\n', size(X,1), size(X,2));

%% =========================================================
% 2. FEATURE SELECTION (Welch T-Test)
%% =========================================================

p_values = zeros(1, size(X,2));

for i = 1:size(X,2)
    [~, p] = ttest2(X(y==0,i), X(y==1,i), 'Vartype','unequal');
    p_values(i) = p;
end

[p_sorted, idx_sorted] = sort(p_values);
Significant_Indices = idx_sorted(p_sorted < 0.05);

fprintf(' Number of significant features: %d\n', numel(Significant_Indices));

%% =========================================================
% 3. SVM LOOP (Increasing Number of Features)
%% =========================================================

Best_Acc   = 0;
Best_K     = 0;
Best_Stats = [];

for k = 1:length(Significant_Indices)

    selected_idx = Significant_Indices(1:k);
    X_sub = X(:, selected_idx);

    MDL = fitcsvm( ...
        X_sub, y, ...
        'Standardize', true, ...
        'KernelFunction','linear', ...
        'KernelScale','auto', ...
        'Prior','uniform', ...
        'LeaveOut','on');

    y_pred = kfoldPredict(MDL);

    cm = confusionmat(y, y_pred);
    TN = cm(1,1); FP = cm(1,2);
    FN = cm(2,1); TP = cm(2,2);

    acc  = (TP+TN)/sum(cm(:));
    sens = TP/(TP+FN);
    spec = TN/(TN+FP);
    prec = TP/(TP+FP);
    f1   = 2*(prec*sens)/(prec+sens);

    if isnan(f1), f1 = 0; end

    if acc > Best_Acc
        Best_Acc   = acc;
        Best_K     = k;
        Best_Stats = [acc sens spec f1];
    end
end

%% =========================================================
% 4. RESULTS
%% =========================================================

ResultTable = table( ...
    "Pearson", ...
    Best_K, ...
    Best_Stats(1)*100, ...
    Best_Stats(2)*100, ...
    Best_Stats(3)*100, ...
    Best_Stats(4)*100, ...
    'VariableNames', { ...
        'Method','Number_of_Features', ...
        'Accuracy','Sensitivity','Specificity','F1_Score'});

fprintf('\n --- BEST RESULT --- \n');
disp(ResultTable);
