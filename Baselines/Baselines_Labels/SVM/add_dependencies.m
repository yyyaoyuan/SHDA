%%%%% addpath %%%%%
addpath('./libsvm-3.22/matlab');
OR = '../../../Datasets/Datasets_Labels/ObjectRecognition';
TI = '../../../Datasets/Datasets_Labels/T2IClassification';
TC = '../../../Datasets/Datasets_Labels/TextCategorization';
addpath(OR);
addpath(TI);
addpath(TC);
%%%%% addname %%%%%
%----------------------------------------------------------------%
% ObjectRecognition Datasets
SAS = 'Source_Amazon_Surf.mat'; 
SCS = 'Source_Caltech_Surf.mat'; 
SWS = 'Source_Webcam_Surf.mat'; 
%------------------------------%
TCD = 'Target_Caltech_Decaf.mat'; 
TDD = 'Target_Dslr_Decaf.mat'; 
TWD = 'Target_Webcam_Decaf.mat'; 
%----------------------------------------------------------------%
% T2IClassification Datasets
%------------------------------%
SNN = 'Source_Nustag_Neural.mat';
TID = 'Target_Imgnet_Decaf.mat';
%----------------------------------------------------------------%
% TextCategorization
SEN = 'Source_EN.mat';
SFR = 'Source_FR.mat';
SGR = 'Source_GR.mat';
SIT = 'Source_IT.mat';
TS5 = 'Target_SP_5.mat';
%----------------------------------------------------------------%