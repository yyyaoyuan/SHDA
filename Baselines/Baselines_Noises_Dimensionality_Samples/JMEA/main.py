import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
# Total
#source_exp = [ad.ND100, ad.ND200, ad.ND300, ad.ND400, ad.ND500, ad.NS300, ad.NS400, ad.NS600, ad.NS700]
#target_exp = [ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5, ad.TS5]
#result = 'JMEA_ND_NS'

source_exp = [ad.ND100_10, ad.ND200_10, ad.ND300_10, ad.ND400_10, ad.ND500_10, ad.NS300_10, ad.NS400_10, ad.NS500_10, ad.NS600_10, ad.NS700_10]
target_exp = [ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD, ad.TCD]
result = 'JMEA_ND_NS_10'


# Test
#source_exp = [ad.ND100]
#target_exp = [ad.TS5]
#result = 'JMEA_ND_NS_test'

Task_num = range(len(source_exp))
# ===========================================================#
# --------------------------------------------------------#
if __name__ == "__main__":
    # parameters
    tf.set_random_seed(1234)
    beta = 0.0001  # control MMD loss 0.001 (good for NUS-WIDE+ImageNet-8 not for Office+Caltech-10 and Multilingual Reuters Collection (0.0001))
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
    acc_jmea_list = multiprocessing.Manager().list()
    acc_jmea = np.zeros((length, iter))

    for i in Task_num:
        acc = 0
        print("Source domain: " + source_exp[i])
        print("Target domain: " + target_exp[i])
        # load data
        source = sio.loadmat(source_exp[i])
        target = sio.loadmat(target_exp[i])
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
            #-------------------------------------#
            
            ns, ds = xs.shape
            nl, dt = xl.shape
            # print(nl)
            nu, _ = xu.shape

            # norm for whole data
            # xs = preprocessing.normalize(xs, norm='l2', axis=0)
            # xt = np.concatenate([xl,xu],0)
            # xt = preprocessing.normalize(xt, norm='l2',axis=0)
            # xl = xt[:nl,:]
            # xu = xt[nl:,:]

            nt = nl + nu
            class_number = len(np.unique(xl_label));

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
            acc_jmea[i][j] = acc_jmea_list[i * iter + j]
            print(np.mean(acc_jmea, axis=0), np.std(acc_jmea, axis=0))
    print(np.mean(acc_jmea, axis=1), np.std(acc_jmea, axis=1))
    np.savetxt('results/' + result + '_JEMA.csv', acc_jmea, delimiter=',')
