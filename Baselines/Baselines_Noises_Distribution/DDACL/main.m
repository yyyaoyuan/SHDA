clear;clc
add_dependencies;
rng(1234);
%==========================================================================
% For Noises_Distribution
%source_exp = {NDG, NDU, NDL};
%target_exp = {TS5, TS5, TS5};
%result = 'DDACL_Distribution_TS5.mat';

% Test
source_exp = {NDG10, NDU10, NDL10};
target_exp = {TCD, TCD, TCD};
result = 'DDACL_Distribution_TCD.mat';
%==========================================================================
% parameters:
%beta = 0.001;
%tau = 0.002;
%lambda = 0.001;
beta = 0.001;
tau = 0.001;
lambda = 0.001;
d = 100; % the dimension of the subspace
T = 5; % the iteration
fprintf('beta = %.4f, tau = %.4f, lambda = %.4f, d = %.2f\n', beta, tau, lambda, d);
%==========================================================================
len = length(source_exp);
% iter = 10;  % for final
iter = 10;     % for programming
%------------------------------------------------------------------------%
acc_DDACL = zeros(len,iter); %Acc for CTF
for i = 1:len
    acc1 = 0; %Acc for CTF
    disp(['Source Domain:' source_exp{i}]);
    disp(['Target Domain:' target_exp{i}]);
    for j = 1:iter
        fprintf('===================itertion[%d]===================\n',j);
        %---------------------------------------------------%
        % loda data
        load(source_exp{i});
        load(target_exp{i});        
        Xl = training_features{j};
        Xl_Label = training_labels{j};
        Xl = normr(Xl);     % get the normalized labeled target data
        Xs = source_features;
        Xs_Label = source_labels;
        Xs = normr(Xs);
        Xu = testing_features{j};
        Xu_Label = testing_labels{j};
        Xu = normr(Xu);
        %---------------------------------------------------%
        % learning
        [Wt,bt,Pt,Ps,VectorObj] = DDA(Xs,Xs_Label,Xl,Xl_Label,Xu,Xu_Label,beta,tau,lambda,d,T);        
        %---------------------------------------------------%
        % prediction
        [~,ft_preLabel_u] = Softmax(Xu,Wt,bt,Pt);   % ft(xu);
        [~,ft_preLabel_l] = Softmax(Xl,Wt,bt,Pt);   % ft(xl);
        [~,ft_preLabel_s] = Softmax(Xs,Wt,bt,Ps);   % ft(xs);
        %---------------------------------------------------%
        ft_acc_u = Evaluate(ft_preLabel_u,Xu_Label)*100;
        ft_acc_l = Evaluate(ft_preLabel_l,Xl_Label)*100;
        ft_acc_s = Evaluate(ft_preLabel_s,Xs_Label)*100;
        %---------------------------------------------------%
        fprintf('ft(xu) accuracy is:%f\n',ft_acc_u);
        fprintf('ft(xl) accuracy is:%f\n',ft_acc_l);
        fprintf('ft(xs) accuracy is:%f\n',ft_acc_s);
        acc_DDACL(i,j) = ft_acc_u;        
    end
    fprintf('===========================================================\n');
    fprintf('DDACL Total Acc:%f, Average_DDACL = %f +/- %f\n',mean(acc_DDACL(i,:)), std(acc_DDACL(i,:))/sqrt(iter));
end
save(result, 'acc_DDACL');
