% set paths
addpath('./libs/libsvm-weights-3.22/matlab');
addpath('./libs/sumkernels');
addpath('./commfuns');
%%%%% addpath %%%%%
ND = '../../../Datasets/Datasets_Noises/Noises_Dimensionality';
NS = '../../../Datasets/Datasets_Noises/Noises_Samples';
TC = '../../../Datasets/Datasets_Labels/TextCategorization';
OR = '../../../Datasets/Datasets_Labels/ObjectRecognition';

addpath(TC);
addpath(OR);
addpath(ND);
addpath(NS);
%%%%% addname %%%%%
%----------------------------------------------------------------%
% Noises_Dimensionality
ND100 = 'Noise_Dim_100.mat';
ND200 = 'Noise_Dim_200.mat';
ND300 = 'Noise_Dim_300.mat';
ND400 = 'Noise_Dim_400.mat';
ND500 = 'Noise_Dim_500.mat';

ND100_10 = 'Noise_Dim_100_10.mat';
ND200_10 = 'Noise_Dim_200_10.mat';
ND300_10 = 'Noise_Dim_300_10.mat';
ND400_10 = 'Noise_Dim_400_10.mat';
ND500_10 = 'Noise_Dim_500_10.mat';
%----------------------------------------------------------------%
% Noises_Samples
NS300 = 'Noise_Sam_300.mat';
NS400 = 'Noise_Sam_400.mat';
NS500 = 'Noise_Sam_500.mat';
NS600 = 'Noise_Sam_600.mat';
NS700 = 'Noise_Sam_700.mat';

NS300_10 = 'Noise_Sam_300_10.mat';
NS400_10 = 'Noise_Sam_400_10.mat';
NS500_10 = 'Noise_Sam_500_10.mat';
NS600_10 = 'Noise_Sam_600_10.mat';
NS700_10 = 'Noise_Sam_700_10.mat';
%----------------------------------------------------------------%
% Target Domain
TS5 = 'Target_SP_5.mat';
TCD = 'Target_Caltech_Decaf.mat';