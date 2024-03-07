clear; clc;
add_dependencies;
rng(1234);
% =============================================
% set params
param.C_s = 1;
param.C_t = 1;
param.C_x = 1e-3;
param.sigma         = 1;
param.mkl_degree    = 1.5;
param.ratio_var     = 0;
param.hfa_iter      = 50;
param.hfa_tau       = 0.001;
%==========================================================================
%Prepare Data (Choose the three setting below by yourself)
%-------------------------------------------------------------------------%
% ObjectRecognition Datasets
% source_exp = {SAS, SCS, SWS};
% target_exp = {TCD, TWD, TDD};
% result = 'SHFA_OR.mat';
%-------------------------------------------------------------------------%
% T2IClassification Datasets
% source_exp = {SNN};
% target_exp = {TID};
% result = 'SHFA_TI.mat';
%-------------------------------------------------------------------------%
% TextCategorization
% source_exp = {SEN, SFR, SGR, SIT};
% target_exp = {TS5, TS5, TS5, TS5};
% result = 'SHFA_TC.mat';
%-------------------------------------------------------------------------%
% Test
source_exp = {SWS};
target_exp = {TDD};
result = 'test.mat';
%==========================================================================
len = length(source_exp);
iter = 1;
%--------------------------------------%
%add by YAO,Yuan
total_SHFA = cell(1,len);

for i = 1:len
    disp(['Source Domain:' source_exp{i}]);
    disp(['Target Domain:' target_exp{i}]);
    
    load(source_exp{i});
    load(target_exp{i});
    %--------------------------------------%
    %len_labels = length(source_labels);
    len_labels = 2;
    %--------------------------------------%
    
    acc1_SHFA = zeros(len_labels, iter); %Acc for SHFA
    
    for k = 1:len_labels
        fprintf('===================iteration[%d]===================\n',k);
        data.source_features            = source_features';
        data.source_labels              = source_labels{k};
        
        for j = 1:iter
            fprintf('===================iteration[%d]===================\n',j);
            
            data.target_labeled_features    = training_features{j}';
            data.target_unlabeled_features  = testing_features{j}';
            data.target_test_features       = testing_features{j}';
            data.target_labeled_labels      = training_labels{j};
            data.target_test_labels         = testing_labels{j};
            data.target_unlabeled_labels    = testing_labels{j};
            data.categories                 = unique(data.source_labels);
            %-------------------------------------------------------------------------%
            % Preprocessing
            data.source_features = normc(data.source_features);
            data.target_labeled_features = normc(data.target_labeled_features);
            data.target_unlabeled_features = normc(data.target_unlabeled_features);
            data.target_test_features = normc(data.target_test_features);
            %-------------------------------------------------------------------------%
            categories  = data.categories;
            dec_values  = zeros(size(data.target_test_features, 2), length(categories));
            source_features     = data.source_features;
            target_features     = [data.target_labeled_features data.target_unlabeled_features];
            
            % =============================================
            % prepare kernels
            kparam.kernel_type =  'gaussian';
            [K_s, param_s] = getKernel(data.source_features, kparam);
            [K_t, param_t] = getKernel(target_features, kparam);
            
            [K_s_root, resnorm_s] = sqrtm(K_s); K_s_root = real(K_s_root);
            [K_t_root, resnorm_t] = sqrtm(K_t); K_t_root = real(K_t_root);
            n_s = size(K_s, 1);
            n_t = size(K_t, 1);
            
            K       = [K_s zeros(n_s, n_t); zeros(n_t, n_s) K_t];
            K_root  = [K_s_root zeros(n_s, n_t); zeros(n_t, n_s) K_t_root];
            
            K_t_root_inv = real(pinv(K_t_root));
            L_t_inv = [zeros(n_s, n_t); eye(n_t)] * K_t_root_inv;
            
            % do kernel decomposition for inference \y
            aug_features    = sqrtm((1+param.sigma)*K+ones(size(K)));
            aug_features    = real(aug_features);
            % =========================================================================
            % train one-versus-all classifiers
            for c = 1:length(categories)
                % fprintf(1, '-- Class %d: %s\n', c, categories{c});
                source_labels       = 2*(data.source_labels == c) - 1;
                target_labels       = 2*(data.target_labeled_labels == c) - 1;
                % -----------------------------
                % set the ratio of positive samples in unlabeld data, which can be
                % eatimated using the labeled samples, or from prior knowledge.
                % In our paper, we used the ground truth ratio as the prior knowledge.
                % The same value is also used in T-SVM for comparison
                ratio               = sum(data.target_test_labels == c)/length(data.target_test_labels);
                param.upper_ratio   = ratio;
                param.lower_ratio   = ratio;
                
                % training
                [model, Us, labels, coefficients, rho, obj] = train_shfa_pnorm(source_labels, target_labels, K, K_root, aug_features, param);
                % testing
                K_test                  = getKernel(data.target_test_features, target_features, param_t);
                dec_values(:, c)        = predict_ifa_semi_kernel(K_test, model, Us, labels, coefficients, rho, K_root, L_t_inv);
            end
            
            % =========================================================================
            % display results
            test_labels         = data.target_test_labels;
            [~, predict_labels] = max(dec_values, [], 2);
            acc     =  sum(predict_labels == test_labels)/length(test_labels);
            fprintf('SHFA accuracy = %f\n', acc);
            acc1_SHFA(k,j) = acc * 100;
        end
        fprintf('===========================================================\n');
        fprintf('SHFA Total Acc:%f, Average_SHFA = %f +/- %f\n', mean(acc1_SHFA(k,:)), std(acc1_SHFA(k,:))/sqrt(iter));
    end
    total_SHFA{1, i} = acc1_SHFA;
end
save(result, 'total_SHFA');
