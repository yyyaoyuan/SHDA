import tensorflow as tf
import numpy as np
import scipy.io as io
import utils
from nnst import nnst
def run_nnst(source_acc_nnst_list, acc_nnst_list, config, config_data):
    with tf.Session() as sess:
        model = nnst(sess=sess, config=config)
        #------------------------------------------#
        xs = config_data['xs']
        xl = config_data['xl']
        xu = config_data['xu']
        lr = config_data['lr']
        xs_label = config_data['xs_label']
        xl_label = config_data['xl_label']
        xu_label = config_data['xu_label']
        T = config_data['T']
        #index = config_data['index']
        class_number = config['class_number']
        nl = config['nl']
        ns = config['ns']
        nu = config['nu']

        yl = tf.reshape(tf.one_hot(xl_label,class_number,on_value=1,off_value=0), [nl, class_number]) # shape: nl, class_number
        yu = tf.reshape(tf.one_hot(xu_label,class_number,on_value=1,off_value=0), [nu, class_number]) # shape: nu, class_number
        ys = tf.reshape(tf.one_hot(xs_label,class_number,on_value=1,off_value=0), [ns, class_number]) # shape: ns, class_number
        
        ys_r, yl_r, yu_r = sess.run([ys, yl, yu])
        train_feed = {model.input_xs: xs, model.input_ys: ys_r, model.input_xl: xl, model.input_yl: yl_r, 
                model.input_xu: xu, model.input_yu: yu_r, model.learning_rate: lr}

        for t in range(T):
        #------------------------------------------#
            # training feature network
            sess.run(model.train_step, feed_dict=train_feed)
            if t % 10 == 0:
                #------------------------------------------#
                xa_acc, xs_acc, xl_acc, xu_acc = sess.run([model.xa_acc, model.xs_acc, model.xl_acc, model.xu_acc], feed_dict=train_feed) # Compute final evaluation on test data
                #loss_f_xa = sess.run(model.loss_f_xa, feed_dict=train_feed)
                loss_f_xs, loss_f_xl = sess.run([model.loss_f_xs, model.loss_f_xl], feed_dict=train_feed)
                print("the accuracy of f(xa) is: " + str(xa_acc))
                print("the accuracy of f(xs) is: " + str(xs_acc))
                print("the accuracy of f(xl) is: " + str(xl_acc))
                print("the accuracy of f(xu) is: " + str(xu_acc))
                print("----------------------------")
                print("the loss_f_xs is: " + str(loss_f_xs))
                print("the loss_f_xl is: " + str(loss_f_xl))
                print("===============================")
        xs_acc = sess.run(model.xs_acc, feed_dict=train_feed)*100 # Get the final accuracy of xu
        xu_acc = sess.run(model.xu_acc, feed_dict=train_feed)*100 # Get the final accuracy of xu
        loss_f_xs, loss_f_xl = sess.run([model.loss_f_xs, model.loss_f_xl], feed_dict=train_feed)
        #loss_f_xa = sess.run(model.loss_f_xa, feed_dict=train_feed) # Get the final accuracy of xu
        print("the accuracy of f(xs) is: " + str(xs_acc))
        print("the accuracy of f(xu) is: " + str(xu_acc))
        print("the loss_f_xs is: " + str(loss_f_xs))
        print("the loss_f_xl is: " + str(loss_f_xl))
        source_acc_nnst_list.append(xs_acc) # record accuracy of xs
        acc_nnst_list.append(xu_acc) # record accuracy of xu