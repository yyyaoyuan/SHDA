clear all;
close all;
clc;
rng(1234);
addpath('bhtsne');


% n = 300; 
% n = 400; 
% n = 500; % the number of each class, default
% n = 600; 
n = 700; 

d = 300;

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
data7 = mvnrnd(mu7, sigma, n);
data8 = mvnrnd(mu8, sigma, n);
data9 = mvnrnd(mu9, sigma, n);
data10 = mvnrnd(mu10, sigma, n);

source_features = [data1; data2; data3; data4; data5; data6; data7; data8; data9; data10];
source_labels = [ones(n,1); ones(n,1)*2; ones(n,1)*3; ones(n,1)*4; ones(n,1)*5; ones(n,1)*6; ones(n,1)*7; ones(n,1)*8; ones(n,1)*9; ones(n,1)*10];

Plot_data_original(source_features, source_labels);
[mappedX, mapping, score] = lda(source_features, source_labels);
[X_new, W, lambda, score1] = LDA1(source_features, source_labels);

% save('Noises_Samples/Noise_Sam_300_10.mat', 'source_features', 'source_labels');
% save('Noises_Samples/Noise_Sam_400_10.mat', 'source_features', 'source_labels');
% save('Noises_Samples/Noise_Sam_500_10.mat', 'source_features', 'source_labels');
% save('Noises_Samples/Noise_Sam_600_10.mat', 'source_features', 'source_labels');
save('Noises_Samples/Noise_Sam_700_10.mat', 'source_features', 'source_labels');