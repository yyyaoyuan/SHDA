import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import multiprocessing
import scipy.io as sio # read .mat files
import numpy as np
from sklearn import preprocessing # Normalization data
import add_dependencies as ad # add some dependencies
from run_nn import run_nn
import pdb
tf.set_random_seed(1234)
#-----------------------------------------#
# read mat files
#-----------------------------------------#
# Total
#target_exp = [ad.TCD, ad.TWD, ad.TDD, ad.TID, ad.TS5]
#result = 'NNt'
#---------------------------------#
target_exp = [ad.TS5]
result = 'test'
#===========================================================#
#--------------------------------------------------------#
if __name__ == "__main__":
    # parameters
    tau = 0.001 # control regularization term, cannot be an integer
    lr = 0.01 # learning rate
    iter_inner = 500 # the iter number
    #===========================================================#
    length = len(target_exp)
    # iter = 10  # for final, run 10 and get average results
    iter = 1 # for test
    acc_nn_list = multiprocessing.Manager().list()
    acc_nn = np.zeros((length,iter))

    for i in range(0,length):
        acc = 0
        print("Target domain: " + target_exp[i])
        for j in range(0,iter):
            print("====================iteration[" + str(j+1) + "]====================")
            #-------------------------------------#
            # load data
            target = sio.loadmat(target_exp[i])

            xl = target['training_features'][0,j] # read labeled target data
            xl = preprocessing.normalize(xl, norm='l2')
            xl_label = target['training_labels'][0,j] - 1 # read labeled target data labels, form 0 start

            xu = target['testing_features'][0,j]  # read unlabeled target data
            xu = preprocessing.normalize(xu, norm='l2')
            xu_label = target['testing_labels'][0,j] - 1  # read unlabeled target data labels, form 0 start

            nl, dt = xl.shape
            nu, _ = xu.shape
            nt = nl + nu
            class_number = len(np.unique(xl_label));
                
            config = {'dt': dt, 'nl': nl, 'nu': nu, 'class_number': class_number, 'tau': tau}
            config_data = {'xl': xl, 'xu': xu, 'lr': lr, 'iter_inner': iter_inner, 
                           'xl_label': xl_label, 'xu_label': xu_label}
            
            p = multiprocessing.Process(target=run_nn, args=(acc_nn_list,config,config_data))
            p.start()
            p.join()
            acc_nn[i][j] = acc_nn_list[i*iter+j]
    print(np.mean(acc_nn, axis=0))
    np.savetxt('results/'+result+'.csv', acc_nn, delimiter = ',')
