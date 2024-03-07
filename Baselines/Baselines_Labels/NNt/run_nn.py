import tensorflow as tf
import utils
from sklearn.manifold import TSNE
from nn import nn
import pdb
def run_nn(acc_nn_list,config,config_data):
    with tf.Session() as sess:
        model = nn(sess=sess, config=config)
        #------------------------------------------#
        xl = config_data['xl']
        xu = config_data['xu']
        lr = config_data['lr']
        xl_label = config_data['xl_label']
        xu_label = config_data['xu_label']
        iter_inner = config_data['iter_inner']
        class_number = config['class_number']
        nl = config['nl']
        nu = config['nu']

        yl = tf.reshape(tf.one_hot(xl_label,class_number,on_value=1,off_value=0), [nl, class_number]) # shape: nl, class_number
        yu = tf.reshape(tf.one_hot(xu_label,class_number,on_value=1,off_value=0), [nu, class_number]) # shape: nu, class_number
        
        yl_r,yu_r = sess.run([yl,yu])
        train_feed = {model.input_xl: xl, model.input_yl: yl_r, model.input_xu: xu, model.input_yu: yu_r, model.learning_rate: lr}
        for ite_number in range(iter_inner):
        #------------------------------------------#
            # training feature network
            sess.run(model.train_step, feed_dict=train_feed)
            if ite_number % 1 == 0:
                print("the total_loss is: " + str(sess.run(model.total_loss, feed_dict=train_feed)))
                print("-------------" + str(ite_number) + "---------------")
                #------------------------------------------#
                xl_acc, xu_acc = sess.run([model.xl_acc, model.xu_acc], feed_dict=train_feed) # Compute final evaluation on test data
                loss_f_xl= sess.run(model.loss_f_xl, feed_dict=train_feed)
                print("the accuracy of f(xl) is: " + str(xl_acc))
                print("the accuracy of f(xu) is: " + str(xu_acc))
                print("----------------------------")
                print("the loss_f_xl is: " + str(loss_f_xl))
                print("===============================")
        xu_acc = sess.run(model.xu_acc, feed_dict=train_feed)*100 # Get the final accuracy of xu
        print("the accuracy of f(xu) is: " + str(xu_acc))
        #--------------------------#
        # vasiual data
        #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
        #all_data = sess.run(model.all_data, feed_dict=train_feed)
        #tsne = tsne.fit_transform(all_data)
        #utils.plot_all_data(tsne, xs_label, xl_label, xu_label)
        #--------------------------#
        acc_nn_list.append(xu_acc) # record accuracy of xu
