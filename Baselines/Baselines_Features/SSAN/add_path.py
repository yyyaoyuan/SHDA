import add_dependencies as ad # add some dependencies

DATASETS = {
    # Object-ImageNet Datasets
    'SAD_8': ad.SAD_8,
    'SAS_8': ad.SAS_8,
    'SCD_8': ad.SCD_8,
    'SCS_8': ad.SCS_8,
    'SWD_8': ad.SWD_8,
    'SWS_8': ad.SWS_8,
    #------------------------------#
    'TID': ad.TID,
    #----------------------------------------------------------------#
    # Object-Text Datasets
    'SAD_6': ad.SAD_6, 
    'SAS_6': ad.SAS_6, 
    'SCD_6': ad.SCD_6, 
    'SCS_6': ad.SCS_6, 
    'SWD_6': ad.SWD_6, 
    'SWS_6': ad.SWS_6,
    #------------------------------# 
    'TS5': ad.TS5,
    #----------------------------------------------------------------#
    # Tag-Text
    'SNN_6': ad.SNN_6,
}
