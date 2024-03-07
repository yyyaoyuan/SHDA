clear all;
close all;
clc;
rng(1234);
addpath('bhtsne');


% n = 300;
% n = 400;
n = 500; % the number of each class, default
% n = 600;
% n = 700;

d = 300;

mu1 = normrnd(0, 1, 1, d);
mu2 = normrnd(0, 1, 1, d);
mu3 = normrnd(0, 1, 1, d);
mu4 = normrnd(0, 1, 1, d);
mu5 = normrnd(0, 1, 1, d);
mu6 = normrnd(0, 1, 1, d);

sigma = eye(d);

data1 = mvnrnd(mu1, sigma, n);
data2 = mvnrnd(mu2, sigma, n);
data3 = mvnrnd(mu3, sigma, n);
data4 = mvnrnd(mu4, sigma, n);
data5 = mvnrnd(mu5, sigma, n);
data6 = mvnrnd(mu6, sigma, n);

source_features = [data1; data2; data3; data4; data5; data6];
source_labels = [ones(n,1); ones(n,1)*2; ones(n,1)*3; ones(n,1)*4; ones(n,1)*5; ones(n,1)*6];

% Plot_data_original(source_features, source_labels);

% save('Datasets_Noises_Samples/Noise_Sam_300.mat', 'source_features', 'source_labels');
% save('Datasets_Noises_Samples/Noise_Sam_400.mat', 'source_features', 'source_labels');
save('Datasets_Noises_Samples/Noise_Sam_400.mat', 'source_features', 'source_labels');
% save('Datasets_Noises_Samples/Noise_Sam_600.mat', 'source_features', 'source_labels');
% save('Datasets_Noises_Samples/Noise_Sam_700.mat', 'source_features', 'source_labels');