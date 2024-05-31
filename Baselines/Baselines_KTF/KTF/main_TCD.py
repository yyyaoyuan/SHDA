import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import multiprocessing
import scipy.io as io # read .mat files
import numpy as np
from sklearn import preprocessing # Normalization data
import add_dependencies as ad # add some dependencies
from run_nn import run_nn
import pdb
import warnings
tf.set_random_seed(1234)
# 忽略所有警告信息
warnings.filterwarnings('ignore')
#-----------------------------------------#
# read mat files
#-----------------------------------------#
target_exp = [ad.TCD]
#===========================================================#
if __name__ == "__main__":
    # parameters
    tau = 0.05 # control regularization term, cannot be an integer
    beta = 0.1 # control f_s
    mu = 1. # control mmd % for TCD 1; for TS5 0.1
    lr = 0.001 # learning rate
    T = 600 # the iter number
    #===========================================================#
    iter = 1 
    
    xl_acc = []
    xu_acc = []
    xs_acc = []

    loss_f_xl = []
    loss_f_xs = []

    loss_mmd = []

    source_path = '../../../Datasets/Datasets_Noises/Noises_DifferentMS_10_100/'
    source_domains = os.listdir(source_path)
    
    for source_domain in source_domains:
        if source_domain.endswith('.mat'):
            source_domain_path = os.path.join(source_path, source_domain)
        else:
            print('This is not a mat file')
            continue

        print("Source domain: " + source_domain_path)
        print("Target domain: " + target_exp[0])

        source = io.loadmat(source_domain_path)
        target = io.loadmat(target_exp[0])
    
        xl_acc_list = multiprocessing.Manager().list()
        xu_acc_list = multiprocessing.Manager().list()
        xs_acc_list = multiprocessing.Manager().list()

        loss_f_xl_list = multiprocessing.Manager().list()
        loss_f_xs_list = multiprocessing.Manager().list()

        loss_mmd_list = multiprocessing.Manager().list()
        #-------------------------------------#
        ps = source['source_features'] # read projected source data
        ps = preprocessing.normalize(ps, norm='l2')
        xs_label = source['source_labels'] - 1 # read source data labels, form 0 start

        for j in range(0,iter):
            print("====================iteration[" + str(j+1) + "]====================")

            xl = target['training_features'][0,j] # read labeled target data
            xl = preprocessing.normalize(xl, norm='l2')
            xl_label = target['training_labels'][0,j] - 1 # read labeled target data labels, form 0 start

            xu = target['testing_features'][0,j]  # read unlabeled target data
            xu = preprocessing.normalize(xu, norm='l2')
            xu_label = target['testing_labels'][0,j] - 1  # read unlabeled target data labels, form 0 start

            nl, dt = xl.shape
            nu, _ = xu.shape
            ns, d = ps.shape
            nt = nl + nu
            C = len(np.unique(xl_label));
                
            config = {'dt': dt, 'nl': nl, 'nu': nu, 'ns': ns, 'C': C, 'd': d, 
                      'tau': tau, 'beta': beta, 'mu': mu}
            config_data = {'xl': xl, 'xu': xu, 'ps': ps, 'lr': lr, 'T': T,  
                           'xl_label': xl_label, 'xu_label': xu_label, 'xs_label': xs_label}
            
            p = multiprocessing.Process(target=run_nn, args=(xl_acc_list, xu_acc_list, xs_acc_list, loss_f_xl_list, loss_f_xs_list, loss_mmd_list, config, config_data))
            p.start()
            p.join()

            xl_acc += xl_acc_list
            xu_acc += xu_acc_list
            xs_acc += xs_acc_list
            loss_f_xl += loss_f_xl_list
            loss_f_xs += loss_f_xs_list
            loss_mmd += loss_mmd_list

    io.savemat('LearnedResults_TCD/xl_acc', {'xl_acc': xl_acc})
    io.savemat('LearnedResults_TCD/xu_acc', {'xu_acc': xu_acc})
    io.savemat('LearnedResults_TCD/xs_acc', {'xs_acc': xs_acc})

    io.savemat('LearnedResults_TCD/loss_f_xl', {'loss_f_xl': loss_f_xl})
    io.savemat('LearnedResults_TCD/loss_f_xs', {'loss_f_xs': loss_f_xs})

    io.savemat('LearnedResults_TCD/loss_mmd', {'loss_mmd': loss_mmd})
