%%%%% addpath %%%%%
addpath('./CDLS_functions/');
addpath('./libsvm-weights-3.22/matlab');
%%%%% addpath %%%%%
ND = '../../../Datasets/Datasets_Noises/Noises_Distribution';
OT = '../../../Datasets/Datasets_Features/Object-Text';
OR = '../../../Datasets/Datasets_Labels/ObjectRecognition';

addpath(ND);
addpath(OT);
addpath(OR);
%%%%% addname %%%%%
NDU = 'Noise_Uniform.mat';
NDU10 = 'Noise_Uniform_10.mat';
NDL = 'Noise_Laplace.mat';
NDL10 = 'Noise_Laplace_10.mat';
NDG = 'Noise_Gaussian.mat';
NDG10 = 'Noise_Gaussian_10.mat';
%----------------------------------------------------------------%
TS5 = 'Target_SP_5.mat';
TCD = 'Target_Caltech_Decaf.mat';
%----------------------------------------------------------------%