clear all;
close all;
clc;
rng(1234);
addpath('bhtsne');
addpath('oldLDA');
n = 100; % the number of each class
d = 300;

%--------------------------------------------------------------------%
% Uniform
%
% a = -10;
% b = 10;
% 
% data1 = unifrnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data2 = unifrnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data3 = unifrnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data4 = unifrnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data5 = unifrnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data6 = unifrnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data7 = unifrnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data8 = unifrnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data9 = unifrnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data10 = unifrnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
%--------------------------------------------------------------------%
% Laplace 
% a = 0;
% b = 1;
% 
% data1 = laprnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data2 = laprnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data3 = laprnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data4 = laprnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data5 = laprnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data6 = laprnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data7 = laprnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data8 = laprnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data9 = laprnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
% data10 = laprnd(a, b, n, d); % 生成 n * d 阶矩阵服从 U(a,b) 分布的随机数
%--------------------------------------------------------------------%
% Gaussian 
mu1 = normrnd(0, 1, 1, d);
mu2 = normrnd(0, 1, 1, d);
mu3 = normrnd(0, 1, 1, d);
mu4 = normrnd(0, 1, 1, d);
mu5 = normrnd(0, 1, 1, d);
mu6 = normrnd(0, 1, 1, d);
mu7 = normrnd(0, 1, 1, d);
mu8 = normrnd(0, 1, 1, d);
mu9 = normrnd(0, 1, 1, d);
mu10 = normrnd(0, 1, 1, d);

sigma = eye(d);

data1 = mvnrnd(mu1, sigma, n);
data2 = mvnrnd(mu2, sigma, n);
data3 = mvnrnd(mu3, sigma, n);
data4 = mvnrnd(mu4, sigma, n);
data5 = mvnrnd(mu5, sigma, n);
data6 = mvnrnd(mu6, sigma, n);
data7 = mvnrnd(mu6, sigma, n);
data8 = mvnrnd(mu6, sigma, n);
data9 = mvnrnd(mu6, sigma, n);
data10 = mvnrnd(mu6, sigma, n);
%--------------------------------------------------------------------%

source_features = [data1; data2; data3; data4; data5; data6; data7; data8; data9; data10];
source_labels = [ones(n,1); ones(n,1)*2; ones(n,1)*3; ones(n,1)*4; ones(n,1)*5; ones(n,1)*6; ones(n,1)*7; ones(n,1)*8; ones(n,1)*9; ones(n,1)*10];

% Plot_data_original(source_features, source_labels);

% save('Noises_Distribution/Noise_Uniform_10.mat', 'source_features', 'source_labels');
% save('Noises_Distribution/Noise_Laplace_10.mat', 'source_features', 'source_labels');
save('Noises_Distribution/Noise_Gaussian_10.mat', 'source_features', 'source_labels');