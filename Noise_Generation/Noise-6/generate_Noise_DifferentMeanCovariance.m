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
name_list = {'Noise_alpha_02', 'Noise_alpha_04', 'Noise_alpha_06', 'Noise_alpha_08', 'Noise_alpha_10'};

% calculated results
% mean_norm_list = [12.6219463555948	24.4367114396530	36.1596192428001	46.7423895673635	60.4254947321968];
% sigma_norm_list = [105.343124369177	210.315865860404	315.392046734035	420.528974673999	525.045955402112];

for i = 1:5

    mu1 = normrnd(0, 1, 1, d) * 1 * alpha_list(1, i);
    mu2 = normrnd(0, 1, 1, d) * 2 * alpha_list(1, i);
    mu3 = normrnd(0, 1, 1, d) * 3 * alpha_list(1, i);
    mu4 = normrnd(0, 1, 1, d) * 4 * alpha_list(1, i);
    mu5 = normrnd(0, 1, 1, d) * 5 * alpha_list(1, i);
    mu6 = normrnd(0, 1, 1, d) * 6 * alpha_list(1, i);

    mean_norm = (norm(mu1) + norm(mu2) + norm(mu3) + norm(mu4) + norm(mu5) + norm(mu6)) ./ 6;
    mean_norm_list(1, i) = mean_norm;

    sigma1 = generateSigma(d) * 1 * alpha_list(1, i);
    sigma2 = generateSigma(d) * 2 * alpha_list(1, i);
    sigma3 = generateSigma(d) * 3 * alpha_list(1, i);
    sigma4 = generateSigma(d) * 4 * alpha_list(1, i);
    sigma5 = generateSigma(d) * 5 * alpha_list(1, i);
    sigma6 = generateSigma(d) * 6 * alpha_list(1, i);

    mean_sigma = (norm(sigma1, 'fro') + norm(sigma2, 'fro') + norm(sigma3, 'fro') + norm(sigma4, 'fro') + norm(sigma5, 'fro') + norm(sigma6, 'fro')) ./ 6;
    sigma_norm_list(1, i) = mean_sigma;

    data1 = mvnrnd(mu1, sigma1, n);
    data2 = mvnrnd(mu2, sigma2, n);
    data3 = mvnrnd(mu3, sigma3, n);
    data4 = mvnrnd(mu4, sigma4, n);
    data5 = mvnrnd(mu5, sigma5, n);
    data6 = mvnrnd(mu6, sigma6, n);

    source_features = [data1; data2; data3; data4; data5; data6];
    source_labels = [ones(n,1); ones(n,1)*2; ones(n,1)*3; ones(n,1)*4; ones(n,1)*5; ones(n,1)*6];

    % Plot_data_original(source_features, source_labels);
%     disp(['Noises_DifferentMeanCovariance/', name_list{1, i} ,'.mat']);
    save(['Noises_DifferentMeanCovariance/', name_list{1, i}, '.mat'], 'source_features', 'source_labels');
end