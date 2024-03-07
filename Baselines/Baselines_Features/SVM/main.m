clear all
add_dependencies;
rng(1234);
%==========================================================================
%Prepare Data (Choose the three setting below by yourself)
% target_exp = {TID,TS5};
% result = 'SVM.mat';
target_exp = {TID};
result = 'test.mat';
%==========================================================================
C = 1;
len = length(target_exp);
iter = 10;
%add by YAO,Yuan
acc1_SVM = zeros(len,iter);  %Acc for SVM
for i = 1:len
    acc1 = 0; %Acc for Baseline
    
    disp(['Target Domain:' target_exp{i}]);
    
    for j = 1:iter
        fprintf('===================iteration[%d]===================\n',j);
        
        load(target_exp{i});
        
        T = training_features{j};
        T_Label = training_labels{j};
        T = normr(T);
        
        Ttest = testing_features{j};
        Ttest_Label = testing_labels{j};
        Ttest = normr(Ttest);
        
        T = T';
        Ttest = Ttest';
        
        %Baseline
        [model] = svmtrain(T_Label,T',['-c ', num2str(C), ' -t 0 -q']);
        [~,acc,b] = svmpredict(Ttest_Label,Ttest',model,'-q');
        fprintf('Baseline Acc:%f\n',acc(1));
        %add bu YAO,Yuan
        acc1_SVM(i,j) = acc(1);
        acc1 = acc1 + acc(1);
        %==========================================================================
    end
    fprintf('===========================================================\n');
    fprintf('SVM Total Acc:%f, Average_SVM = %f +/- %f\n',acc1/iter, mean(acc1_SVM(i,:)), std(acc1_SVM(i,:))/sqrt(iter));
end
save(result, 'acc1_SVM');