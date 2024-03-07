import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
import multiprocessing
import scipy.io as sio # read .mat files
import numpy as np
from sklearn import preprocessing # Normalization data
import add_dependencies as ad # add some dependencies
from run_nnst import run_nnst
import pdb
tf.set_random_seed(1234)
#-----------------------------------------#
# read mat files
#-----------------------------------------#
# Noises_Dimensionality && Noises_Samples
source_exp = [ad.ND100, ad.ND200, ad.ND300, ad.ND400, ad.ND500, ad.NS300, ad.NS400, ad.NS500, ad.NS600, ad.NS700]
target_exp = [ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5]
result = 'HCN_DS_TS5'

#source_exp = [ad.ND100_10, ad.ND200_10, ad.ND300_10, ad.ND400_10, ad.ND500_10, ad.NS300_10, ad.NS400_10, ad.NS500_10, ad.NS600_10, ad.NS700_10]
#target_exp = [ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD]
#result = 'HCN_DS_TCD'

# Noises_Distribution
# source_exp = [ad.NDG, ad.NDU, ad.NDL, ad.NDG10, ad.NDU10, ad.NDL10]
# target_exp = [ad.TS5, ad.TS5, ad.TS5, ad.TCD, ad.TCD, ad.TCD]
# result = 'HCN_Distribution_TS5_TCD'

# Features
# source_exp = [ad.SAS_8, ad.SCS_8, ad.SWS_8, ad.SAD_8, ad.SCD_8, ad.SWD_8, ad.SAS_6, ad.SCS_6, ad.SWS_6, ad.SAD_6, ad.SCD_6, ad.SWD_6, ad.SNN_6]
# target_exp = [ad.TID, ad.TID, ad.TID, ad.TID, ad.TID, ad.TID, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5]
# result = 'Features'

# Labels
# source_exp = [ad.SAS, ad.SCS, ad.SWS, ad.SEN, ad.SFR, ad.SGR, ad.SIT, ad.SNN]
# target_exp = [ad.TCD, ad.TWD, ad.TDD, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TID]
# result = 'Labels'
#===========================================================#
#--------------------------------------------------------#
if __name__ == "__main__":
    # parameters
    lr = 0.001 # learning rate
    T = 200 # the total iter number
    d = 256 # the dimension of common subspace
    tau = 0.005 # control regularization term, cannot be an integer
    beta = 0.1 # control the importance of risk_xs
#===========================================================#
    length = len(source_exp)
    # iter = 10  # for final, run 10 and get average results
    iter = 10 # for test

    source_acc_nnst_list = multiprocessing.Manager().list()
    source_acc_nnst = np.zeros((length, iter))
    
    acc_nnst_list = multiprocessing.Manager().list()
    acc_nnst = np.zeros((length, iter))
    
    for i in range(0,length):
        print("Source domain: " + source_exp[i])
        print("Target domain: " + target_exp[i])
        for j in range(0,iter):
            print("====================iteration[" + str(j+1) + "]====================")
            #-------------------------------------#
            # load data
            source = sio.loadmat(source_exp[i])
            target = sio.loadmat(target_exp[i])

            xl = target['training_features'][0,j] # read labeled target data
            xl = preprocessing.normalize(xl, norm='l2')
            xl_label = target['training_labels'][0,j] - 1 # read labeled target data labels, form 0 start

            xu = target['testing_features'][0,j]  # read unlabeled target data
            xu = preprocessing.normalize(xu, norm='l2')
            xu_label = target['testing_labels'][0,j] - 1  # read unlabeled target data labels, form 0 start
            
            xs = source['source_features'] # read source data
            xs_label = source['source_labels'] - 1 # read source data labels, form 0 start
            xs = preprocessing.normalize(xs, norm='l2')

            ns, ds = xs.shape
            nl, dt = xl.shape
            nu, _ = xu.shape
            nt = nl + nu
            class_number = len(np.unique(xl_label))
                
            config = {'ds': ds, 'dt': dt, 'ns': ns, 'nl': nl, 'nu': nu, 'class_number': class_number, 
                      'd': d, 'tau': tau, 'beta': beta}
            config_data = {'xs': xs, 'xl': xl, 'xu': xu, 'lr': lr, 'T': T, 
                           'xs_label': xs_label, 'xl_label': xl_label, 'xu_label': xu_label}
            
            p = multiprocessing.Process(target=run_nnst, args=(source_acc_nnst_list, acc_nnst_list, config, config_data))
            p.start()
            p.join()
            source_acc_nnst[i][j] = source_acc_nnst_list[i*iter+j]
            acc_nnst[i][j] = acc_nnst_list[i*iter+j]
    print(np.mean(source_acc_nnst, axis=1))
    print(np.mean(acc_nnst, axis=1))
    np.savetxt('results/'+result+'_source_acc.csv', source_acc_nnst, delimiter = ',')
    np.savetxt('results/'+result+'_acc.csv', acc_nnst, delimiter = ',')
