%% =========================================================
%  ADIM 3: SVM TABANLI ÖZELLİK SEÇİMİ ve DEĞERLENDİRME
%% =========================================================

clear; clc; close all;

%% =========================================================
% 1. VERİ YÜKLEME
%% =========================================================


klasor_veri = fullfile(pwd, 'ALZHEIMER_SVM_HON_SPEARMAN');
dosya_yolu  = fullfile(klasor_veri, 'Alzheimer_HonSpearman_ML.mat');

if ~exist(dosya_yolu, 'file')
    error(' Veri dosyası bulunamadı!');
end

load(dosya_yolu);   % -> SVM_Veri gelir

X = SVM_Veri.X;   % (N x 400)
y = SVM_Veri.y;   % (N x 1)


fprintf(' Veri yüklendi: %d örnek, %d özellik\n', size(X,1), size(X,2));

%% =========================================================
% 2. ÖZNİTELİK SEÇİMİ (Welch T-Test)
%% =========================================================

p_degerleri = zeros(1, size(X,2));

for i = 1:size(X,2)
    [~, p] = ttest2(X(y==0,i), X(y==1,i), 'Vartype','unequal');
    p_degerleri(i) = p;
end

[p_sirali, idx_sirali] = sort(p_degerleri);
Anlamli_Indeksler = idx_sirali(p_sirali < 0.05);

fprintf(' Anlamlı özellik sayısı: %d\n', numel(Anlamli_Indeksler));

%% =========================================================
% 3. SVM DÖNGÜSÜ (Artan Özellik Sayısı)
%% =========================================================

En_Iyi_Acc   = 0;
En_Iyi_K     = 0;
En_Iyi_Stats = [];

for k = 1:length(Anlamli_Indeksler)

    secilen_idx = Anlamli_Indeksler(1:k);
    X_alt = X(:, secilen_idx);

    MDL = fitcsvm( ...
        X_alt, y, ...
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

    if acc > En_Iyi_Acc
        En_Iyi_Acc   = acc;
        En_Iyi_K     = k;
        En_Iyi_Stats = [acc sens spec f1];
    end
end

%% =========================================================
% 4. SONUÇLAR
%% =========================================================

SonucTablosu = table( ...
    "Pearson", ...
    En_Iyi_K, ...
    En_Iyi_Stats(1)*100, ...
    En_Iyi_Stats(2)*100, ...
    En_Iyi_Stats(3)*100, ...
    En_Iyi_Stats(4)*100, ...
    'VariableNames', { ...
        'Yontem','Ozellik_Sayisi', ...
        'Accuracy','Sensitivity','Specificity','F1_Score'});

fprintf('\n --- EN İYİ SONUÇ --- \n');
disp(SonucTablosu);
