clear;clc
add_dependencies;
rng(1234);
%==========================================================================
% For TS5
%source_exp = {NDG, NDU, NDL};
%target_exp = {TS5, TS5, TS5};
%result = 'TNT_Distribution_TS5.mat';

% For TCD
source_exp = {NDG10, NDU10, NDL10};
target_exp = {TCD, TCD, TCD};
result = 'TNT_Distribution_TCD.mat';
%==========================================================================
len = length(source_exp);
iter = 10;
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
