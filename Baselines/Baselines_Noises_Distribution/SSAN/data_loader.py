import torch
import os.path as osp
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import add_path as ap
from utils import seed_everything

def get_configuration(args, index):
    seed_everything(args)
    SOURCE_PATH = ap.DATASETS[args.source]
    TARGET_PATH = ap.DATASETS[args.target]

    # source and target domain infos
    print('========= Source & Target Info =========')
    print('Source Domain: ' + SOURCE_PATH)
    print('Target Domain: ' + TARGET_PATH)
    print('========= Loading Data =========')
    source = sio.loadmat(SOURCE_PATH)
    target = sio.loadmat(TARGET_PATH)
    print('========= Loading Data Completed =========')
    print()
    print('========= Data Information =========')

    xl = target['training_features'][0, index] # read labeled target data
    xl = preprocessing.normalize(xl, norm='l2')
    xl_label = target['training_labels'][0, index] - 1 # read labeled target data labels, form 0 start

    xu = target['testing_features'][0, index]  # read unlabeled target data
    xu = preprocessing.normalize(xu, norm='l2')
    xu_label = target['testing_labels'][0, index] - 1  # read unlabeled target data labels, form 0 start
           
    xs = source['source_features'] # read source data
    xs = preprocessing.normalize(xs, norm='l2')   
    xs_label = source['source_labels'] - 1 # read source data labels, form 0 start

    class_number = len(np.unique(xs_label))  # number of classes

    ns, ds = xs.shape  # ns = number of source instances, ds = dimension of source instances
    nl, dt = xl.shape  # nl = number of labeled target instances, ds = dimension of all target instances
    nu, _ = xu.shape
    nt = nl + nu  # total amount of target instances
    print('ns = ', ns)
    print('nl = ', nl)
    print('nu = ', nu)
    print('ds = ', ds)
    print('dt = ', dt)
    print('class_number: ', class_number)
    print()

    # Generate dataset objects
    source_data = [torch.from_numpy(xs), torch.from_numpy(xs_label)]
    labeled_target_data = [torch.from_numpy(xl), torch.from_numpy(xl_label)]
    unlabeled_target_data = [torch.from_numpy(xu), torch.from_numpy(xu_label)]

    # Data Allocation In Each Batch
    print('Number of Source Instances: ' + str(ns))
    print('Number of Labeled Target Instances: ' + str(nl))
    print('Number of Unlabeled Target Instances: ' + str(nu))
    print()

    # data configurations
    configuration = {'ns': ns, 'nl': nl, 'nu': nu, 'nt': nt, 'class_number': class_number,
                     'd_source': ds, 'd_target': dt,
                     'source_data': source_data, 'labeled_target_data': labeled_target_data,
                     'unlabeled_target_data': unlabeled_target_data}

    print('========= Loading Done =========')
    print()
    print('========= Training Started =========')
    return configuration
