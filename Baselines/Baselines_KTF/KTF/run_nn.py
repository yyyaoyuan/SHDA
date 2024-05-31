import tensorflow as tf
import utils
from sklearn.manifold import TSNE
from nn import nn
import pdb
def run_nn(xl_acc_list, xu_acc_list, xs_acc_list, loss_f_xl_list, loss_f_xs_list, loss_mmd_list, config, config_data):
    with tf.Session() as sess:
        model = nn(sess=sess, config=config)
        #------------------------------------------#
        xl = config_data['xl']
        xu = config_data['xu']
        ps = config_data['ps']
        lr = config_data['lr']
        xl_label = config_data['xl_label']
        xu_label = config_data['xu_label']
        xs_label = config_data['xs_label']
        T = config_data['T']
        C = config['C']
        nl = config['nl']
        nu = config['nu']
        ns = config['ns']

        yl = tf.reshape(tf.one_hot(xl_label,C, on_value=1, off_value=0), [nl, C]) # shape: nl, C
        yu = tf.reshape(tf.one_hot(xu_label,C, on_value=1, off_value=0), [nu, C]) # shape: nu, C
        ys = tf.reshape(tf.one_hot(xs_label,C, on_value=1, off_value=0), [ns, C]) # shape: ns, C
        
        yl_r, yu_r, ys_r = sess.run([yl, yu, ys])
        train_feed = {model.xl: xl, model.yl: yl_r, model.xu: xu, model.yu: yu_r, model.ps: ps, model.ys: ys_r, model.lr: lr}
        #------------------------------------------#
        for k in range(T):
            # training feature network
            sess.run(model.train_step, feed_dict=train_feed)
            
            xl_acc, xu_acc, xs_acc = sess.run([model.xl_acc, model.xu_acc, model.xs_acc], feed_dict=train_feed) # Compute final evaluation on test data
            loss_f_xl, loss_f_xs, loss_mmmd, loss_cmmd, loss_mmd, loss_reg = sess.run([model.loss_f_xl, model.loss_f_xs, model.loss_mmmd, model.loss_cmmd, model.loss_mmd, model.loss_reg], feed_dict=train_feed)
            
            xl_acc_list.append(xl_acc) # record accuracy of xl
            xu_acc_list.append(xu_acc) # record accuracy of xu
            xs_acc_list.append(xs_acc) # record accuracy of xs

            loss_f_xl_list.append(loss_f_xl) # record loss_f_xl
            loss_f_xs_list.append(loss_f_xs) # record loss_f_xs

            loss_mmd_list.append(loss_mmd) # record loss_mmd

            if k % 1 == 0:
                print("-------------" + str(k) + "---------------")
                #------------------------------------------#
                print("the accuracy of f(xl) is: " + str(xl_acc))
                print("the accuracy of f(xu) is: " + str(xu_acc))
                print("the accuracy of f(xs) is: " + str(xs_acc))
                print("----------------------------")
                print("the loss_f_xl is: " + str(loss_f_xl))
                print("the loss_f_xs is: " + str(loss_f_xs))
                print("the loss_mmd is: " + str(loss_mmd))
        
        print("Final results")
        print("the loss_f_xl is: " + str(loss_f_xl))
        print("the loss_f_xs is: " + str(loss_f_xs))
        print("the loss_mmd is: " + str(loss_mmd))
