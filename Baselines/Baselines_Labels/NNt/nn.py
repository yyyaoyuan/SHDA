import tensorflow as tf
import numpy as np
import add_dependencies as ad # add some dependencies
import utils
import pdb

class nn(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.dt = config['dt']
        self.nl = config['nl']
        self.nu = config['nu']
        self.class_number = config['class_number']
        self.tau = config['tau']
        self.nt = self.nl+self.nu
        self.create_model()

    def create_model(self):
        #-----------------------------------------#        
        with tf.name_scope('inputs'):
            self.input_xl = tf.placeholder(tf.float32, [None, self.dt], name='input_xl')
            self.input_yl = tf.placeholder(tf.int32, [None, self.class_number], name='input_yl')
            self.input_xu = tf.placeholder(tf.float32, [None, self.dt], name='input_xu')
            self.input_yu = tf.placeholder(tf.int32, [None, self.class_number], name='input_yu')
            self.learning_rate = tf.placeholder(tf.float32, [], name='lr')
            self.input_xt = tf.concat([self.input_xl, self.input_xu], 0, name='input_xt')
        #------------------------------------------#
        # build classifier networks of f
        h = 256
        f_w1 = tf.Variable(tf.truncated_normal([self.dt, h], stddev=0.01))
        f_w2 = tf.Variable(tf.truncated_normal([h, self.class_number], stddev=0.01))
        f_b1 = tf.Variable(tf.truncated_normal([h], stddev=0.01))
        f_b2 = tf.Variable(tf.truncated_normal([self.class_number], stddev=0.01))

        self.f_h_logits = utils.add_layer(self.input_xt, f_w1, f_b1, tf.nn.leaky_relu)
        self.f_xt_logits = utils.add_layer(self.f_h_logits, f_w2, f_b2, None)
        self.f_xl_logits = tf.slice(self.f_xt_logits, [0, 0], [self.nl, -1]) # extract f_xl_logists from f_xt_logists
        self.f_xu_logits = tf.slice(self.f_xt_logits, [self.nl, 0], [self.nu, -1]) # extract f_xu_logists from f_x_logists
        self.pseudo_yu = tf.nn.softmax(self.f_xu_logits)
        #------------------------------------------#
        with tf.name_scope('loss_f_xl'):
            self.loss_f_xl = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_yl, logits=self.f_xl_logits))
        #------------------------------------------#
        # reguralization term
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(f_w1))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(f_w2))
        self.reg = tf.add_n(tf.get_collection("loss"))
        #------------------------------------------#
        with tf.name_scope('total_loss'):
            self.total_loss = self.loss_f_xl + self.reg
        # the accuracy of xl
        pred_xl = tf.nn.softmax(self.f_xl_logits)
        correct_pred_xl = tf.equal(tf.argmax(self.input_yl,1), tf.argmax(pred_xl,1))
        self.xl_acc = tf.reduce_mean(tf.cast(correct_pred_xl, tf.float32))
        # the accuracy of xu
        correct_pred_xu = tf.equal(tf.argmax(self.input_yu,1), tf.argmax(self.pseudo_yu,1))
        self.xu_acc = tf.reduce_mean(tf.cast(correct_pred_xu, tf.float32))
        #------------------------------------------#
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
        #self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_loss)
        #self.train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.05).minimize(self.total_loss)

        #writer = tf.summary.FileWriter("log/", self.sess.graph)
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.sess.run(init)

