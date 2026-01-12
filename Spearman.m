
clear; clc; close all;

%% =========================================================
% 1. GENEL AYARLAR
%% =========================================================

AnaKlasor = '/Users/haticekumru/Desktop/Alzheimer_fnets';
CiktiKlasoru = fullfile(pwd, 'ALZHEIMER_ML_VERISI_Spearman');
if ~exist(CiktiKlasoru, 'dir'), mkdir(CiktiKlasoru); end

% ROI seçimi (410 → 400)
ROI_Indeksleri = [1:200, 211:410];
ROI_Sayisi = numel(ROI_Indeksleri);

% Bağlantı parametreleri
Esik_Orani = 0.26;   % En güçlü %26 bağlantıyı tut
GuvenliAtanh = @(x) atanh(max(min(x,0.999999),-0.999999));

%% =========================================================
% 2. VERİ SETİ TANIMI
%% =========================================================

Gruplar = {'45_CN_yeni', '45_AD_yeni'};
Etiketler = [0, 1];   % 0: Sağlıklı (CN), 1: Alzheimer (AD)

X_Tum = [];          % (N x 400 x 400) → GNN 
X_MLP = [];          % (N x 400) → MLP
y_Tum = [];

fprintf(' Veri işleme başlıyor...\n');

%% =========================================================
% 3. ANA VERİ İŞLEME DÖNGÜSÜ
%% =========================================================

for g = 1:numel(Gruplar)
    
    GrupYolu = fullfile(AnaKlasor, Gruplar{g});
    Dosyalar = dir(fullfile(GrupYolu, '*.txt'));
    
    fprintf('  %s grubu (%d dosya)\n', Gruplar{g}, numel(Dosyalar));
    
    for i = 1:numel(Dosyalar)
        try
            %% --- A) VERİ OKUMA ---
            Dosya = fullfile(GrupYolu, Dosyalar(i).name);
            Sinyal = readmatrix(Dosya);

            % Zaman x ROI formatını garanti et
            if size(Sinyal,1) < size(Sinyal,2)
                Sinyal = Sinyal';
            end

            %% --- B) ROI KESME ---
            if size(Sinyal,2) < 410, continue; end
            Sinyal = Sinyal(:, ROI_Indeksleri);

            %% --- C) PEARSON FC HESABI ---
            FC = corr(Sinyal,'Type','Spearman');
            FC(isnan(FC)) = 0;
            FC(1:ROI_Sayisi+1:end) = 0;  % Köşegen sıfır

            %% --- D) EŞİKLEME ---
            ust_ucgen = abs(FC(triu(true(ROI_Sayisi),1)));
            esik = prctile(ust_ucgen, (1-Esik_Orani)*100);
            FC(abs(FC) < esik) = 0;

            %% --- E) FISHER-Z ---
            FC_Z = GuvenliAtanh(FC);

            %% --- F) KAYIT (GNN / DeepSet) ---
            X_Tum = cat(3, X_Tum, FC_Z);
            y_Tum = [y_Tum; Etiketler(g)];

            %% --- G) ÖZNİTELİK (MLP) ---
            NodeStrength = mean(abs(FC_Z),2)';
            X_MLP = [X_MLP; NodeStrength];

        catch
            continue;
        end
    end
end

%% =========================================================
% 4. VERİ TEMİZLİĞİ
%% =========================================================

X_MLP(isnan(X_MLP)) = 0;
ROI_Etiketleri = compose("ROI_%03d",1:ROI_Sayisi)';

%% =========================================================
% 5. VERİ KAYDI (CLASSICAL ML)
%% =========================================================

SVM_Veri.X   = X_MLP;        % (N x 400)
SVM_Veri.y   = y_Tum;        % (N x 1)
SVM_Veri.ROI = ROI_Etiketleri;

save(fullfile(CiktiKlasoru,'Alzheimer_Spearman_ML.mat'),'SVM_Veri');

fprintf(' Classical ML (SVM) verisi kaydedildi\n');




