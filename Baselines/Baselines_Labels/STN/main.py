import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
import multiprocessing
import scipy.io as sio # read .mat files
import numpy as np
from sklearn import preprocessing # Normalization data
import add_dependencies as ad # add some dependencies
from run_stn import run_stn
import pdb
tf.set_random_seed(1234)
#-----------------------------------------#
# read mat files
#-----------------------------------------#
# For ObjectRecognition Datasets
# source_exp = [ad.SAS, ad.SCS, ad.SWS]
# target_exp = [ad.TCD, ad.TWD, ad.TDD]
# result = 'STN_OR'

# For T2IClassification Datasets
# source_exp = [ad.SNN]
# target_exp = [ad.TID]
# result = 'STN_TI'

# For TextCategorization Datasets
# source_exp = [ad.SEN, ad.SFR, ad.SGR, ad.SIT]
# target_exp = [ad.TS5, ad.TS5, ad.TS5, ad.TS5]
# result = 'STN_TC'
#-----------------------------------------#
# Test
source_exp = [ad.SAS]
target_exp = [ad.TCD]
result = 'test'
#===========================================================#
#--------------------------------------------------------#
if __name__ == "__main__":
    # parameters
    beta = 0.001 # control MMD loss
    lr = 0.001 # learning rate
    T = 300 # the total iter number
    T1 = 0
    T2 = T
    d = 256 # the dimension of common subspace
    tau = 0.001 # control regularization term, cannot be an integer    
#===========================================================#
    length = len(source_exp)
    # iter = 10  # for final, run 10 and get average results
    iter = 1 # for test
    total_STN = np.empty((1,length),dtype=object)

    for i in range(0,length):
        acc = 0
        print("Source domain: " + source_exp[i])
        print("Target domain: " + target_exp[i])
        # load data
        source = sio.loadmat(source_exp[i])
        target = sio.loadmat(target_exp[i])

        xs = source['source_features'] # read source data
        xs_label_all = source['source_labels']
        xs = preprocessing.normalize(xs, norm='l2')
        _, len_labels = xs_label_all.shape
        acc_stn_list = multiprocessing.Manager().list()
        acc_stn = np.zeros((len_labels,iter))
        
        for k in range(0, len_labels):
            print("---------------------xs_label_all[" + str(k+1) + "]---------------------")
            xs_label = xs_label_all[0, k] - 1 # read source data labels, form 0 start
           
            for j in range(0,iter):
                print("====================iteration[" + str(j+1) + "]====================")
                
                xl = target['training_features'][0,j] # read labeled target data
                xl = preprocessing.normalize(xl, norm='l2')
                xl_label = target['training_labels'][0,j] - 1 # read labeled target data labels, form 0 start

                xu = target['testing_features'][0,j]  # read unlabeled target data
                xu = preprocessing.normalize(xu, norm='l2')
                xu_label = target['testing_labels'][0,j] - 1  # read unlabeled target data labels, form 0 start

                ns, ds = xs.shape
                nl, dt = xl.shape
                nu, _ = xu.shape
                nt = nl + nu
                class_number = len(np.unique(xl_label))
                
                config = {'ds': ds, 'dt': dt, 'ns': ns, 'nl': nl, 'nu': nu, 'class_number': class_number, 'beta': beta, 'tau': tau, 'd': d}
                config_data = {'xs': xs, 'xl': xl, 'xu': xu, 'lr': lr, 'T': T, 'T1': T1, 'T2': T2, 
                           'xs_label': xs_label, 'xl_label': xl_label, 'xu_label': xu_label}
            
                p = multiprocessing.Process(target=run_stn, args=(acc_stn_list,config,config_data))
                p.start()
                p.join()
                acc_stn[k][j] = acc_stn_list[k*iter+j]
        print(np.mean(acc_stn, axis=1))
        total_STN[0, i] = acc_stn
    # np.savetxt('results/'+result+'.csv', acc_stn, delimiter = ',')
    sio.savemat('results/'+result+'.mat', {'total_STN': total_STN})






