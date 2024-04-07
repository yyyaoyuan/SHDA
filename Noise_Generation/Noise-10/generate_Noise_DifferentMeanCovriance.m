clear all;
close all;
clc;
rng(123);
addpath('bhtsne');
n = 100; % the number of each class
d = 300;

mean_norm_list = zeros(1,5);
sigma_norm_list = zeros(1,5);

alpha_list = [0.2, 0.4, 0.6, 0.8, 1.0];
name_list = {'Noise_alpha_02_10', 'Noise_alpha_04_10', 'Noise_alpha_06_10', 'Noise_alpha_08_10', 'Noise_alpha_10_10'};

% calculated results
% mean_norm_list = [19.4511005446701	38.4264968747712	57.2390307989067	77.0220314084880	95.5394478738253];
% sigma_norm_list = [164.803381448762	330.818490058521	496.160430774706	661.603724690199	824.461903734606];

for i = 1:5

    mu1 = normrnd(0, 1, 1, d) * 1 * alpha_list(1, i);
    mu2 = normrnd(0, 1, 1, d) * 2 * alpha_list(1, i);
    mu3 = normrnd(0, 1, 1, d) * 3 * alpha_list(1, i);
    mu4 = normrnd(0, 1, 1, d) * 4 * alpha_list(1, i);
    mu5 = normrnd(0, 1, 1, d) * 5 * alpha_list(1, i);
    mu6 = normrnd(0, 1, 1, d) * 6 * alpha_list(1, i);
    mu7 = normrnd(0, 1, 1, d) * 7 * alpha_list(1, i);
    mu8 = normrnd(0, 1, 1, d) * 8 * alpha_list(1, i);
    mu9 = normrnd(0, 1, 1, d) * 9 * alpha_list(1, i);
    mu10 = normrnd(0, 1, 1, d) * 10 * alpha_list(1, i);

    mean_norm = (norm(mu1) + norm(mu2) + norm(mu3) + norm(mu4) + norm(mu5) + norm(mu6) + norm(mu7) + norm(mu8) + norm(mu9) + norm(mu10)) ./ 10;
    mean_norm_list(1, i) = mean_norm;

    sigma1 = generateSigma(d) * 1 * alpha_list(1, i);
    sigma2 = generateSigma(d) * 2 * alpha_list(1, i);
    sigma3 = generateSigma(d) * 3 * alpha_list(1, i);
    sigma4 = generateSigma(d) * 4 * alpha_list(1, i);
    sigma5 = generateSigma(d) * 5 * alpha_list(1, i);
    sigma6 = generateSigma(d) * 6 * alpha_list(1, i);
    sigma7 = generateSigma(d) * 7 * alpha_list(1, i);
    sigma8 = generateSigma(d) * 8 * alpha_list(1, i);
    sigma9 = generateSigma(d) * 9 * alpha_list(1, i);
    sigma10 = generateSigma(d) * 10 * alpha_list(1, i);

    mean_sigma = (norm(sigma1, 'fro') + norm(sigma2, 'fro') + norm(sigma3, 'fro') + norm(sigma4, 'fro') + norm(sigma5, 'fro') ...
                + norm(sigma6, 'fro') + norm(sigma7, 'fro') + norm(sigma8, 'fro') + norm(sigma9, 'fro') + norm(sigma10, 'fro')) ./ 10;
    sigma_norm_list(1, i) = mean_sigma;

    data1 = mvnrnd(mu1, sigma1, n);
    data2 = mvnrnd(mu2, sigma2, n);
    data3 = mvnrnd(mu3, sigma3, n);
    data4 = mvnrnd(mu4, sigma4, n);
    data5 = mvnrnd(mu5, sigma5, n);
    data6 = mvnrnd(mu6, sigma6, n);
    data7 = mvnrnd(mu7, sigma7, n);
    data8 = mvnrnd(mu8, sigma8, n);
    data9 = mvnrnd(mu9, sigma9, n);
    data10 = mvnrnd(mu10, sigma10, n);

    source_features = [data1; data2; data3; data4; data5; data6; data7; data8; data9; data10];
    source_labels = [ones(n,1); ones(n,1)*2; ones(n,1)*3; ones(n,1)*4; ones(n,1)*5; ones(n,1)*6; ones(n,1)*7; ones(n,1)*8; ones(n,1)*9; ones(n,1)*10];

    % Plot_data_original(source_features, source_labels);
%     disp(['Noises_DifferentMeanCovariance/', name_list{1, i} ,'.mat']);
    save(['Noises_DifferentMeanCovariance/', name_list{1, i}, '.mat'], 'source_features', 'source_labels');
end