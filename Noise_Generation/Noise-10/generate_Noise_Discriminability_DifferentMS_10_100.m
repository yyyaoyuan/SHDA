clear all;
close all;
clc;
rng(1234);
addpath('bhtsne');
d = 256;

index = 1;
for alpha = 0.05:0.1:10

    n = randi([100, 1000], 1, 10);
    disp(n);

    mu1 = normrnd(0, 1, 1, d) * 1 * alpha;
    mu2 = normrnd(0, 1, 1, d) * 2 * alpha;
    mu3 = normrnd(0, 1, 1, d) * 3 * alpha;
    mu4 = normrnd(0, 1, 1, d) * 4 * alpha;
    mu5 = normrnd(0, 1, 1, d) * 5 * alpha;
    mu6 = normrnd(0, 1, 1, d) * 6 * alpha;
    mu7 = normrnd(0, 1, 1, d) * 7 * alpha;
    mu8 = normrnd(0, 1, 1, d) * 8 * alpha;
    mu9 = normrnd(0, 1, 1, d) * 9 * alpha;
    mu10 = normrnd(0, 1, 1, d) * 10 * alpha;

    sigma1 = generateSigma(d) * 1 * alpha;
    sigma2 = generateSigma(d) * 2 * alpha;
    sigma3 = generateSigma(d) * 3 * alpha;
    sigma4 = generateSigma(d) * 4 * alpha;
    sigma5 = generateSigma(d) * 5 * alpha;
    sigma6 = generateSigma(d) * 6 * alpha;
    sigma7 = generateSigma(d) * 7 * alpha;
    sigma8 = generateSigma(d) * 8 * alpha;
    sigma9 = generateSigma(d) * 9 * alpha;
    sigma10 = generateSigma(d) * 10 * alpha;

    data1 = mvnrnd(mu1, sigma1, n(1));
    data2 = mvnrnd(mu2, sigma2, n(2));
    data3 = mvnrnd(mu3, sigma3, n(3));
    data4 = mvnrnd(mu4, sigma4, n(4));
    data5 = mvnrnd(mu5, sigma5, n(5));
    data6 = mvnrnd(mu6, sigma6, n(6));
    data7 = mvnrnd(mu7, sigma7, n(7));
    data8 = mvnrnd(mu8, sigma8, n(8));
    data9 = mvnrnd(mu9, sigma9, n(9));
    data10 = mvnrnd(mu10, sigma10, n(10));

    source_features = [data1; data2; data3; data4; data5; data6; data7; data8; data9; data10];
    source_labels = [ones(n(1),1); ones(n(2),1)*2; ones(n(3),1)*3; ones(n(4),1)*4; ones(n(5),1)*5; ones(n(6),1)*6; ones(n(7),1)*7; ones(n(8),1)*8; ones(n(9),1)*9; ones(n(10),1)*10];

%     Plot_data_original(source_features, source_labels);
    
    save(['Noises_DifferentMS_10_100/Noise_index_', num2str(index), '.mat'], 'source_features', 'source_labels');
    index = index + 1;

end
