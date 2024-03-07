#----------------------------------------------------------------
# Features of source samples
OI = '../../../Datasets/Datasets_Features/Object-ImageNet'
OT = '../../../Datasets/Datasets_Features/Object-Text'
TT = '../../../Datasets/Datasets_Features/Tag-Text'

#----------------------------------------------------------------
# Labels of source samples
TC = '../../../Datasets/Datasets_Labels/TextCategorization'
OR = '../../../Datasets/Datasets_Labels/ObjectRecognition'
TI = '../../../Datasets/Datasets_Labels/T2IClassification'

#----------------------------------------------------------------
# Noises
ND = '../../../Datasets/Datasets_Noises/Noises_Dimensionality'
NS = '../../../Datasets/Datasets_Noises/Noises_Samples'
NDT = '../../../Datasets/Datasets_Noises/Noises_Distribution'

#----------------------------------------------------------------
# Noises_Dimensionality
ND100 = ND + '/Noise_Dim_100.mat'
ND200 = ND + '/Noise_Dim_200.mat'
ND300 = ND + '/Noise_Dim_300.mat'
ND400 = ND + '/Noise_Dim_400.mat'
ND500 = ND + '/Noise_Dim_500.mat'

ND100_10 = ND + '/Noise_Dim_100_10.mat'
ND200_10 = ND + '/Noise_Dim_200_10.mat'
ND300_10 = ND + '/Noise_Dim_300_10.mat'
ND400_10 = ND + '/Noise_Dim_400_10.mat'
ND500_10 = ND + '/Noise_Dim_500_10.mat'
#----------------------------------------------------------------
# Noises_Samples
NS300 = NS + '/Noise_Sam_300.mat'
NS400 = NS + '/Noise_Sam_400.mat'
NS500 = NS + '/Noise_Sam_500.mat'
NS600 = NS + '/Noise_Sam_600.mat'
NS700 = NS + '/Noise_Sam_700.mat'

NS300_10 = NS + '/Noise_Sam_300_10.mat'
NS400_10 = NS + '/Noise_Sam_400_10.mat'
NS500_10 = NS + '/Noise_Sam_500_10.mat'
NS600_10 = NS + '/Noise_Sam_600_10.mat'
NS700_10 = NS + '/Noise_Sam_700_10.mat'
#----------------------------------------------------------------
# Distributions
NDU = NDT + '/Noise_Uniform.mat'
NDL = NDT + '/Noise_Laplace.mat'
NDG = NDT + '/Noise_Gaussian.mat'
NDU10 = NDT + '/Noise_Uniform_10.mat'
NDL10 = NDT + '/Noise_Laplace_10.mat'
NDG10 = NDT + '/Noise_Gaussian_10.mat'
#----------------------------------------------------------------
# Target_Domain
TS5 = TC + '/Target_SP_5.mat'
TCD = OR + '/Target_Caltech_Decaf.mat'
TID = TI + '/Target_Imgnet_Decaf.mat'
#----------------------------------------------------------------
# ObjectRecognition Datasets
SAS = OR + '/Source_Amazon_Surf.mat'
SCS = OR + '/Source_Caltech_Surf.mat'
SWS = OR + '/Source_Webcam_Surf.mat'
#------------------------------
TCD = OR + '/Target_Caltech_Decaf.mat'
TDD = OR + '/Target_Dslr_Decaf.mat'
TWD = OR + '/Target_Webcam_Decaf.mat'
#----------------------------------------------------------------
# T2IClassification Datasets
#------------------------------
SNN = TI + '/Source_Nustag_Neural.mat'
TID = TI + '/Target_Imgnet_Decaf.mat'
#----------------------------------------------------------------
# TextCategorization
SEN = TC + '/Source_EN.mat'
SFR = TC + '/Source_FR.mat'
SGR = TC + '/Source_GR.mat'
SIT = TC + '/Source_IT.mat'
TS5 = TC + '/Target_SP_5.mat'
#----------------------------------------------------------------
# Object-ImageNet Datasets
SAD_8 = OI + '/Source_Amazon_Decaf_8.mat'
SAS_8 = OI + '/Source_Amazon_Surf_8.mat'
SCD_8 = OI + '/Source_Caltech_Decaf_8.mat'
SCS_8 = OI + '/Source_Caltech_Surf_8.mat'
SWD_8 = OI + '/Source_Webcam_Decaf_8.mat'
SWS_8 = OI + '/Source_Webcam_Surf_8.mat'
#------------------------------
TID = OI + '/Target_Imgnet_Decaf.mat'
#----------------------------------------------------------------
# Object-Text Datasets
#------------------------------
SAD_6 = OT + '/Source_Amazon_Decaf_6.mat'
SAS_6 = OT + '/Source_Amazon_Surf_6.mat'
SCD_6 = OT + '/Source_Caltech_Decaf_6.mat'
SCS_6 = OT + '/Source_Caltech_Surf_6.mat'
SWD_6 = OT + '/Source_Webcam_Decaf_6.mat'
SWS_6 = OT + '/Source_Webcam_Surf_6.mat'
TS5 = OT + '/Target_SP_5.mat'
#----------------------------------------------------------------
# Tag-Text
SNN_6 = TT + '/Source_Nustag_Neural_6.mat'
