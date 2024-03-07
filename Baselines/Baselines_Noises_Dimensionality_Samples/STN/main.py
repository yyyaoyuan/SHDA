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
#source_exp = [ad.ND100, ad.ND200, ad.ND300, ad.ND400, ad.ND500, ad.NS300, ad.NS400, ad.NS600, ad.NS700]
#target_exp = [ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5]
#result = 'STN_ND_NS'

source_exp = [ad.ND100_10, ad.ND200_10, ad.ND300_10, ad.ND400_10, ad.ND500_10, ad.NS300_10, ad.NS400_10, ad.NS500_10, ad.NS600_10, ad.NS700_10]
target_exp = [ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD]
result = 'STN_ND_NS_10'

# Test
#source_exp = [ad.ND100]
#target_exp = [ad.TS5]
#result = 'STN_ND_NS_test'
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
    iter = 10 # for test
    acc_stn_list = multiprocessing.Manager().list()
    acc_stn = np.zeros((length,iter))

    for i in range(0,length):
        acc = 0
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
                
            config = {'ds': ds, 'dt': dt, 'ns': ns, 'nl': nl, 'nu': nu, 'class_number': class_number, 'beta': beta, 'tau': tau, 'd': d}
            config_data = {'xs': xs, 'xl': xl, 'xu': xu, 'lr': lr, 'T': T, 'T1': T1, 'T2': T2, 
                           'xs_label': xs_label, 'xl_label': xl_label, 'xu_label': xu_label}
            
            p = multiprocessing.Process(target=run_stn, args=(acc_stn_list,config,config_data))
            p.start()
            p.join()
            acc_stn[i][j] = acc_stn_list[i*iter+j]
    print(np.mean(acc_stn, axis=1))
    np.savetxt('results/'+result+'.csv', acc_stn, delimiter = ',')
