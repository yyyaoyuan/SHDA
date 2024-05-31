clear all;
close all;
clc;
rng(1234);
addpath('bhtsne');
d = 256;

index = 1;
for alpha = 0.05:0.1:10

    n = randi([100, 1000], 1, 6);
    disp(n);

    mu1 = normrnd(0, 1, 1, d) * 1 * alpha;
    mu2 = normrnd(0, 1, 1, d) * 2 * alpha;
    mu3 = normrnd(0, 1, 1, d) * 3 * alpha;
    mu4 = normrnd(0, 1, 1, d) * 4 * alpha;
    mu5 = normrnd(0, 1, 1, d) * 5 * alpha;
    mu6 = normrnd(0, 1, 1, d) * 6 * alpha;

    M = [mu1; mu2; mu3; mu4; mu5; mu6];

    sigma1 = generateSigma(d) * 1 * alpha;
    sigma2 = generateSigma(d) * 2 * alpha;
    sigma3 = generateSigma(d) * 3 * alpha;
    sigma4 = generateSigma(d) * 4 * alpha;
    sigma5 = generateSigma(d) * 5 * alpha;
    sigma6 = generateSigma(d) * 6 * alpha;

    data1 = mvnrnd(mu1, sigma1, n(1));
    data2 = mvnrnd(mu2, sigma2, n(2));
    data3 = mvnrnd(mu3, sigma3, n(3));
    data4 = mvnrnd(mu4, sigma4, n(4));
    data5 = mvnrnd(mu5, sigma5, n(5));
    data6 = mvnrnd(mu6, sigma6, n(6));

    source_features = [data1; data2; data3; data4; data5; data6];
    source_labels = [ones(n(1),1); ones(n(2),1)*2; ones(n(3),1)*3; ones(n(4),1)*4; ones(n(5),1)*5; ones(n(6),1)*6];

%     Plot_data_original(source_features, source_labels);
    
    save(['Noises_DifferentMS_6_100/Noise_index_', num2str(index), '.mat'], 'source_features', 'source_labels');
    index = index + 1;
    
end
