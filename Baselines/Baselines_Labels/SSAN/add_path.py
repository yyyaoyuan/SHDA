import add_dependencies as ad # add some dependencies

DATASETS = {
    #----------------------------------------------------------------
    # ObjectRecognition Datasets
    'SAS': ad.SAS,
    'SCS': ad.SCS,
    'SWS': ad.SWS,
    #------------------------------#
    'TCD': ad.TCD,
    'TDD': ad.TDD,
    'TWD': ad.TWD,
    #----------------------------------------------------------------
    # T2IClassification Datasets
    #------------------------------
    'SNN': ad.SNN,
    'TID': ad.TID,
    #----------------------------------------------------------------
    # TextCategorization
    'SEN': ad.SEN,
    'SFR': ad.SFR,
    'SGR': ad.SGR,
    'SIT': ad.SIT,
    'TS5': ad.TS5,
}