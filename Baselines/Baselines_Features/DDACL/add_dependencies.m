%%%%% addpath %%%%%
OI = '../../../Datasets/Datasets_Features/Object-ImageNet';
OT = '../../../Datasets/Datasets_Features/Object-Text';
TT = '../../../Datasets/Datasets_Features/Tag-Text';
addpath(OI);
addpath(OT);
addpath(TT);
%%%%% addname %%%%%
%----------------------------------------------------------------%
% Object-ImageNet Datasets
SAD_8 = 'Source_Amazon_Decaf_8.mat'; 
SAS_8 = 'Source_Amazon_Surf_8.mat'; 
SCD_8 = 'Source_Caltech_Decaf_8.mat'; 
SCS_8 = 'Source_Caltech_Surf_8.mat'; 
SWD_8 = 'Source_Webcam_Decaf_8.mat'; 
SWS_8 = 'Source_Webcam_Surf_8.mat'; 
%------------------------------%
TID = 'Target_Imgnet_Decaf.mat';
%----------------------------------------------------------------%
% Object-Text Datasets
%------------------------------%
SAD_6 = 'Source_Amazon_Decaf_6.mat'; 
SAS_6 = 'Source_Amazon_Surf_6.mat'; 
SCD_6 = 'Source_Caltech_Decaf_6.mat'; 
SCS_6 = 'Source_Caltech_Surf_6.mat'; 
SWD_6 = 'Source_Webcam_Decaf_6.mat'; 
SWS_6 = 'Source_Webcam_Surf_6.mat'; 
TS5 = 'Target_SP_5.mat';
%----------------------------------------------------------------%
% Tag-Text
SNN_6 = 'Source_Nustag_Neural_6.mat';
%----------------------------------------------------------------%