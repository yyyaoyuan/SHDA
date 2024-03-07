import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import multiprocessing
import scipy.io as sio  # read .mat files
from sklearn import preprocessing  # Normalization data
import numpy as np
import add_dependencies as ad  # add some dependencies
from run_J import run_J
#-----------------------------------------#
# read mat files
#-----------------------------------------#
# For ObjectRecognition Datasets
# source_exp = [ad.SAS, ad.SCS, ad.SWS]
# target_exp = [ad.TCD, ad.TWD, ad.TDD]
# result = 'JMEA_OR'

# For T2IClassification Datasets
# source_exp = [ad.SNN]
# target_exp = [ad.TID]
# result = 'JMEA_TI'

# For TextCategorization Datasets
# source_exp = [ad.SEN, ad.SFR, ad.SGR, ad.SIT]
# target_exp = [ad.TS5, ad.TS5, ad.TS5, ad.TS5]
# result = 'JMEA_TC'
#-----------------------------------------#
# Test
source_exp = [ad.SAS]
target_exp = [ad.TCD]
result = 'JMEA_test'


# ===========================================================#
# --------------------------------------------------------#
if __name__ == "__main__":
    # parameters
    tf.set_random_seed(1234)
    beta = 0.001  # control MMD loss 0.001 (good for cifar10 not for cifar100 (0.00001))
    lr = 0.001  # learning rate 0.001 for almost cases
    T = 300  # the total iter number 300 for normal
    T1 = 0
    T2 = T
    d = 256# the dimension of common subspace 256 for almost cases
    tau = 0.001  # control regularization term, cannot be an integer
    startk = 100  # 100 for most cases, 150 for S2D
    # ===========================================================#
    length = len(source_exp)
    iter = 10
    total_JMEA = np.empty((1, length),dtype=object)


    # acc_jmea = np.zeros((length, iter))

    for i in range(0, length):
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
        
        acc_jmea_list = multiprocessing.Manager().list()
        acc_jmea = np.zeros((len_labels, iter))

        for k in range(0, len_labels):
            print("---------------------xs_label_all[" + str(k+1) + "]---------------------")
            xs_label = xs_label_all[0, k] - 1 # read source data labels, form 0 start
        
        
            for j in range(0,iter):
                print("====================iteration[" + str(j+1) + "]====================")
                #-------------------------------------#
                xl = target['training_features'][0,j] # read labeled target data
                xl = preprocessing.normalize(xl, norm='l2')
                xl_label = target['training_labels'][0,j] - 1 # read labeled target data labels, form 0 start

                xu = target['testing_features'][0,j]  # read unlabeled target data
                xu = preprocessing.normalize(xu, norm='l2')
                xu_label = target['testing_labels'][0,j] - 1  # read unlabeled target data labels, form 0 start
                
                #-------------------------------------#
                ns, ds = xs.shape
                nl, dt = xl.shape
                # print(nl)
                nu, _ = xu.shape

            
                nt = nl + nu
                class_number = len(np.unique(xl_label))

                config = {'ds': ds, 'dt': dt, 'ns': ns, 'nl': nl, 'nu': nu, 'class_number': class_number, 'beta': beta,
                          'tau': tau, 'd': d, 'startk': startk}
                config_data = {'xs': xs, 'xl': xl, 'xu': xu, 'lr': lr, 'T': T, 'T1': T1, 'T2': T2,
                               'xs_label': xs_label, 'xl_label': xl_label, 'xu_label': xu_label}
                tf.set_random_seed(1234)
                # run_stn(acc_jmea_list, config, config_data)
                p = multiprocessing.Process(target=run_J, args=(acc_jmea_list, config, config_data))
                p.start()
                p.join()
                print(acc_jmea_list)
                acc_jmea[k][j] = acc_jmea_list[k * iter + j]
        print(np.mean(acc_jmea, axis=1), np.std(acc_jmea, axis=1))
        total_JMEA[0, i] = acc_jmea
    # np.savetxt('results/' + result + '_JEMME_test.csv', acc_jmea, delimiter=',')
    sio.savemat('results/'+result+'.mat', {'total_JMEA': total_JMEA})










