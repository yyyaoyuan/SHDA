clear;clc
add_dependencies;
rng(1234);
%Prepare Data (Choose the three setting below by yourself)
%==========================================================================
%Prepare Data (Choose the three setting below by yourself)
%-------------------------------------------------------------------------%
% ObjectRecognition Datasets
% source_exp = {SAS, SCS, SWS};
% target_exp = {TCD, TWD, TDD};
% result = 'CDLS_OR.mat';
%-------------------------------------------------------------------------%
% T2IClassification Datasets
% source_exp = {SNN};
% target_exp = {TID};
% result = 'CDLS_TI.mat';
%-------------------------------------------------------------------------%
% TextCategorization
% source_exp = {SEN, SFR, SGR, SIT};
% target_exp = {TS5, TS5, TS5, TS5};
% result = 'CDLS_TC.mat';
%-------------------------------------------------------------------------%
% Test
source_exp = {SWS};
target_exp = {TDD};
result = 'test.mat';
%==========================================================================
len = length(source_exp);
iter = 10;
%--------------------------------------%
%add by YAO,Yuan
total_CDLS = cell(1,len);

for i = 1:len
    disp(['Source Domain:' source_exp{i}]);
    disp(['Target Domain:' target_exp{i}]);
    
    load(source_exp{i});
    load(target_exp{i});
    
    %--------------------------------------%
    len_labels = length(source_labels);
    %len_labels = 1;
    %--------------------------------------%
    
    acc1_CDLS = zeros(len_labels, iter); %Acc for CDLS
    
    for k = 1:len_labels
        fprintf('===================iteration[%d]===================\n',k);
        S = source_features;
        S_Label = source_labels{k};
        S = normr(S);
        
        for j = 1:iter
            fprintf('===================iteration[%d]===================\n',j);
            
            T = training_features{j};
            T_Label = training_labels{j};
            T = normr(T);
            
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
            acc1_CDLS(k,j) = acc;
        end
        fprintf('===========================================================\n');
        fprintf('CDLS Total Acc:%f, Average_CDLS = %f +/- %f\n', mean(acc1_CDLS(k,:)), std(acc1_CDLS(k,:))/sqrt(iter));
    end
    total_CDLS{1, i} = acc1_CDLS;
end
save(result, 'total_CDLS');