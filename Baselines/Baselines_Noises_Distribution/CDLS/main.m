clear;clc
add_dependencies;
rng(1234);
%Prepare Data (Choose the three setting below by yourself)
%==========================================================================
% For TS5
%source_exp = {NDG, NDU, NDL};
%target_exp = {TS5, TS5, TS5};
%result = 'CDLS_Distribution_TS5.mat';

% For TCD
source_exp = {NDG10, NDU10, NDL10};
target_exp = {TCD, TCD, TCD};
result = 'CDLS_Distribution_TCD.mat';
%==========================================================================
len = length(source_exp);
iter = 10;
%add by YAO,Yuan
acc1_CDLS = zeros(len, iter); %Acc for GJDA

for i = 1:len
    acc1 = 0; %Acc for CDLS
    
    disp(['Source Domain:' source_exp{i}]);
    disp(['Target Domain:' target_exp{i}]);
    
    for j = 1:iter
        fprintf('===================iteration[%d]===================\n',j);
        
        load(source_exp{i});
        load(target_exp{i});
        
        T = training_features{j};
        T_Label = training_labels{j};
        T = normr(T);
        
        S = source_features;
        S_Label = source_labels;
        S = normr(S);
        
        Ttest = testing_features{j};
        Ttest_Label = testing_labels{j};
        Ttest = normr(Ttest);
        %CDLS
        fprintf('========================CDLS========================\n')
        Data.T = T';
        Data.Ttest = Ttest';
        Data.S = S';
        Data.T_Label = T_Label;
        Data.S_Label = S_Label;
        Data.Ttest_Label = Ttest_Label;
        num_of_T_per_class = length(T_Label)/length(unique(T_Label));
        num_of_L_per_class = length(S_Label)/length(unique(S_Label));
        %%%%% Parameter Setting %%%%%
        param.iter = 5;
        param.scale = num_of_T_per_class/num_of_L_per_class;
        param.delta = 0.5; %% You can tune the portion of the weights if you like (0 < delta <= 1 )
        param.PCA_dimension = 100; %% Make sure this dim. is smaller the source-domain dim.
        
        %%%%% Start CDLS %%%%%
        [acc,alpha,beta] = CDLS(Data,param);
        fprintf('CDLS accuracy is:%f\n',acc);
        %add bu YAO,Yuan
        acc1_CDLS(i,j) = acc;
        acc1 = acc1 + acc;
    end
    fprintf('===========================================================\n');
    fprintf('CDLS Total Acc:%f, Average_CDLS = %f +/- %f\n',acc1/iter, mean(acc1_CDLS(i,:)), std(acc1_CDLS(i,:))/sqrt(iter));
end
save(result, 'acc1_CDLS');
