%% ================= FINAL SVR MODEL WITH FEATURE IMPORTANCE =================
clc;
clear;
close all;

%% STEP 1: Load Dataset
filePath = 'C:\Users\Suganya\Desktop\new paper 1606\2602\FinalDataset.xlsx';
T = readtable(filePath,'VariableNamingRule','modify');

disp('Available Variables:')
disp(T.Properties.VariableNames')

%% STEP 2: Define Target Variables
target_CO = 'CO__ppm';
target_PM = 'PM2_5_ug_m3';

%% STEP 3: Create Derived Features

T.WindTraffic_Interaction = ...
    T.WindSpeed_Micro_m_s_ .* T.Traffic_Count_veh_hr_;

T.Temp_Delta = T.Temp_Micro__C_ - T.Temp_Macro__C_;

T.Wind_Delta = T.WindSpeed_Micro_m_s_ - T.WindSpeed_Macro_m_s_;

%% STEP 4: Select Feature Columns

numericVars = varfun(@isnumeric,T,'OutputFormat','uniform');
allVars = T.Properties.VariableNames;

excludeVars = {'Timestamp','Location_Zone','TimePeriod',...
               target_CO,target_PM};

features = allVars(numericVars & ~ismember(allVars,excludeVars));

disp('Final Feature Set:')
disp(features')

fprintf('Total features used: %d\n',length(features));

%% Labels for plotting
feature_labels = {
'Micro Wind Speed'
'Micro Temperature'
'Micro Humidity'
'Traffic Count'
'Macro Wind Speed'
'Macro Temperature'
'Macro Humidity'
'Wind–Traffic Interaction'
'Temperature Difference'
'Wind Difference'
};

%% STEP 5: Define Target Pollutants
targets = {
    target_CO, 'CO2',  'CO2 Concentration';
    target_PM, 'PM25', 'PM2.5 Concentration'
};

%% ================= MODEL LOOP =================
for i = 1:size(targets,1)

    target_col = targets{i,1};
    short_name = targets{i,2};
    plot_title = targets{i,3};

    %% Prepare Feature Matrix
    X = zeros(height(T), numel(features));

    for j = 1:numel(features)

        col = T.(features{j});
        col(~isfinite(col)) = median(col(~isnan(col)));
        X(:,j) = col;

    end

    %% Target Vector
    Y = T.(target_col);
    Y(~isfinite(Y)) = median(Y(~isnan(Y)));

    %% ================= CROSS VALIDATION =================
    k = 5;

    if strcmp(target_col,target_PM)
        edges = quantile(Y,[0 0.33 0.66 1]);
        Y_bins = discretize(Y,edges);
        cv = cvpartition(Y_bins,'KFold',k);
    else
        cv = cvpartition(height(T),'KFold',k);
    end

    R2_all = zeros(k,1);
    MAE_all = zeros(k,1);
    RMSE_all = zeros(k,1);
    r_all = zeros(k,1);

    for fold = 1:k

        Xtrain = X(training(cv,fold),:);
        Ytrain = Y(training(cv,fold));

        Xtest = X(test(cv,fold),:);
        Ytest = Y(test(cv,fold));

        model = fitrsvm(Xtrain,Ytrain,...
            'KernelFunction','gaussian',...
            'Standardize',true,...
            'KernelScale','auto',...
            'BoxConstraint',1,...
            'Epsilon',0.1);

        Y_pred = predict(model,Xtest);

        SSres = sum((Ytest - Y_pred).^2);
        SStot = sum((Ytest - mean(Ytest)).^2);

        R2_all(fold) = 1 - SSres/SStot;
        MAE_all(fold) = mean(abs(Ytest - Y_pred));
        RMSE_all(fold) = sqrt(mean((Ytest - Y_pred).^2));
        r_all(fold) = corr(Ytest,Y_pred);

    end

    %% ================= PRINT RESULTS =================
    fprintf('\n===== %s =====\n',plot_title)

    fprintf('R2   = %.4f ± %.4f\n',mean(R2_all),std(R2_all))
    fprintf('MAE  = %.4f ± %.4f\n',mean(MAE_all),std(MAE_all))
    fprintf('RMSE = %.4f ± %.4f\n',mean(RMSE_all),std(RMSE_all))
    fprintf('r    = %.4f ± %.4f\n',mean(r_all),std(r_all))

    %% ================= FINAL MODEL =================
    finalModel = fitrsvm(X,Y,...
        'KernelFunction','gaussian',...
        'Standardize',true,...
        'KernelScale','auto',...
        'BoxConstraint',1,...
        'Epsilon',0.1);

    Y_final = predict(finalModel,X);

    %% Save predictions for CFD comparison
    if strcmp(short_name,'CO2')
        Y_final_CO2 = Y_final;
    elseif strcmp(short_name,'PM25')
        Y_final_PM25 = Y_final;
    end

    %% ================= PREDICTION PLOT =================
    figure('Color','w')

    scatter(Y,Y_final,25,'filled')
    hold on
    plot([min(Y) max(Y)], [min(Y) max(Y)], 'r--','LineWidth',2)

    xlabel(['Observed ' plot_title],'FontWeight','bold')
    ylabel(['Predicted ' plot_title],'FontWeight','bold')

    title(['SVR Prediction for ' plot_title],'FontWeight','bold')

    grid on
    set(gca,'FontSize',12)

    saveas(gcf,['SVR_' short_name '_Prediction.png'])

    %% ================= FEATURE IMPORTANCE =================

    baseline_pred = predict(finalModel,X);
    baseline_rmse = sqrt(mean((Y - baseline_pred).^2));

    importance = zeros(length(features),1);

    for f = 1:length(features)

        X_temp = X;
        X_temp(:,f) = X_temp(randperm(size(X_temp,1)),f);

        pred_temp = predict(finalModel,X_temp);

        rmse_temp = sqrt(mean((Y - pred_temp).^2));

        importance(f) = rmse_temp - baseline_rmse;

    end

    importance = importance ./ sum(importance);

    figure('Color','w')

    bar(importance)

    set(gca,'XTick',1:length(features))
    set(gca,'XTickLabel',feature_labels)

    xtickangle(45)

    ylabel('Normalized Feature Importance','FontWeight','bold')
    xlabel('Environmental and Traffic Predictor Variables','FontWeight','bold')

    title(['Permutation-Based Feature Importance for ' plot_title],'FontWeight','bold')

    grid on
    set(gca,'FontSize',11)

    saveas(gcf,['FeatureImportance_' short_name '.png'])

end

%% ================= EXPORT FOR CFD =================

writematrix(Y_final_CO2,'ML_CO2_prediction.csv')
writematrix(Y_final_PM25,'ML_PM25_prediction.csv')

disp('SVR modelling and feature importance analysis completed successfully.')