clear;clc
add_dependencies;
rng(1234);
%==========================================================================
% TS5
%-----------%
% source_exp = {ND100, ND200, ND300, ND400, ND500, ...
%               NS300, NS400, NS600, NS700};
% target_exp = {TS5, TS5, TS5, TS5, TS5, ...
%               TS5, TS5, TS5, TS5};
% result = 'DDACL_Noise_Dim_Sam_TS5.mat';
%-------------------------------------------------------%
% TCD
%-----------%
% source_exp = {ND100_10, ND200_10, ND300_10, ND400_10, ND500_10, ...
%               NS300_10, NS400_10, NS500_10, NS600_10, NS700_10};
% target_exp = {TCD, TCD, TCD, TCD, TCD, ...
%               TCD, TCD, TCD, TCD, TCD};
% result = 'DDACL_Noise_Dim_Sam_TCD.mat';

% Test
source_exp = {ND100};
target_exp = {TS5};
result = 'TNT_ND100TS5.mat';
%==========================================================================
len = length(source_exp);
iter = 1;
%add by YAO,Yuan
acc1_TNT = zeros(len,iter);

for i = 1:len
    acc1 = 0; %Acc for TNT
    disp(['Source Domain:' source_exp{i}]);
    disp(['Target Domain:' target_exp{i}]);
    S.dataset = source_exp{i};
    L.dataset = target_exp{i};
    U.dataset = target_exp{i};
    
    for j = 1:iter
        fprintf('===================iteration[%d]===================\n',j);
        
        load(source_exp{i});
        load(target_exp{i});
        
        T = training_features{j};
        T_Label = training_labels{j};
        T = normr(T);
        
        S1 = source_features;
        S1_Label = source_labels;
        S1 = normr(S1);
        
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
        acc1_TNT(i,j) = acc * 100;
        acc1 = acc1 + acc;
    end
    fprintf('===========================================================\n');
    fprintf('TNT Total Acc:%f, Average_TNT = %f +/- %f\n',acc1/iter, mean(acc1_TNT(i,:)), std(acc1_TNT(i,:))/sqrt(iter));
end
save(result, 'acc1_TNT');