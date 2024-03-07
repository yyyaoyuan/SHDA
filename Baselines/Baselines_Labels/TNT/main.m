clear;clc;
add_dependencies;
rng(1234);
%==========================================================================
%Prepare Data (Choose the three setting below by yourself)
%-------------------------------------------------------------------------%
% ObjectRecognition Datasets
% source_exp = {SAS, SCS, SWS};
% target_exp = {TCD, TWD, TDD};
% result = 'TNT_OR.mat';
%-------------------------------------------------------------------------%
% T2IClassification Datasets
% source_exp = {SNN};
% target_exp = {TID};
% result = 'TNT_TI.mat';
%-------------------------------------------------------------------------%
% TextCategorization
% source_exp = {SEN, SFR, SGR, SIT};
% target_exp = {TS5, TS5, TS5, TS5};
% result = 'TNT_TC.mat';
%-------------------------------------------------------------------------%
% Test
source_exp = {SWS};
target_exp = {TDD};
result = 'test.mat';
%==========================================================================
len = length(source_exp);
iter = 10;
%-----------------------------%
%add by YAO,Yuan
total_TNT = cell(1,len);

for i = 1:len
    disp(['Source Domain:' source_exp{i}]);
    disp(['Target Domain:' target_exp{i}]);
    S.dataset = source_exp{i};
    L.dataset = target_exp{i};
    U.dataset = target_exp{i};
    
    load(source_exp{i});
    load(target_exp{i});
    
    %--------------------------------------%
    len_labels = length(source_labels);
    %len_labels = 1;
    %--------------------------------------%
    
    acc1_TNT = zeros(len_labels,iter); %Acc for TNT
    for k = 1:len_labels
        fprintf('===================iteration[%d]===================\n',k);
        S1 = source_features;
        S1_Label = source_labels{k};
        S1 = normr(S1);
        
        for j = 1:iter
            fprintf('===================iteration[%d]===================\n',j);
            
            T = training_features{j};
            T_Label = training_labels{j};
            T = normr(T);
            
            Ttest = testing_features{j};
            Ttest_Label = testing_labels{j};
            Ttest = normr(Ttest);
            %TNT
            fprintf('========================TNT========================\n')
            S.data = S1';
            S.label = S1_Label';
            L.data = T';
            L.label = T_Label';
            U.data = Ttest';
            U.label = Ttest_Label';
            
            %%%%% Start TNT %%%%%
            acc = TNTforHDA(S,L,U);
            fprintf('TNT accuracy is:%f\n',acc);
            %add bu YAO,Yuan
            acc1_TNT(k,j) = acc * 100;
        end
        fprintf('===========================================================\n');
        fprintf('TNT Total Acc:%f, Average_TNT = %f +/- %f\n', mean(acc1_TNT(k,:)), std(acc1_TNT(k,:))/sqrt(iter));
    end
    total_TNT{1, i} = acc1_TNT;
end
save(result, 'total_TNT');