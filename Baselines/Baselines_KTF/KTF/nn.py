import tensorflow as tf
import numpy as np
import add_dependencies as ad # add some dependencies
import utils
import pdb

class nn(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.dt = config['dt']
        self.d = config['d']
        self.nl = config['nl']
        self.nu = config['nu']
        self.ns = config['ns']
        self.C = config['C']
        self.tau = config['tau']
        self.beta = config['beta']
        self.mu = config['mu']
        self.create_model()

    def create_model(self):
        #-----------------------------------------#        
        with tf.name_scope('inputs'):
            self.xl = tf.placeholder(tf.float32, [None, self.dt], name='xl')
            self.yl = tf.placeholder(tf.int32, [None, self.C], name='yl')
            self.xu = tf.placeholder(tf.float32, [None, self.dt], name='xu')
            self.yu = tf.placeholder(tf.int32, [None, self.C], name='yu')
            self.ps = tf.placeholder(tf.float32, [None, self.d], name='ps') # projected xs
            self.ys = tf.placeholder(tf.int32, [None, self.C], name='ys')
            self.lr = tf.placeholder(tf.float32, [], name='lr')
        #------------------------------------------#
        self.nt = self.nl + self.nu
        self.xt = tf.concat([self.xl, self.xu], 0, name='xt')
        #------------------------------------------#
        # network parameters
        self.d = 256
        #------------------#
        # projector
        f_w1 = tf.Variable(tf.truncated_normal([self.dt, self.d], stddev=0.01))
        f_b1 = tf.Variable(tf.truncated_normal([self.d], stddev=0.01))
        #------------------#
        # classifier
        f_w2 = tf.Variable(tf.truncated_normal([self.d, self.C], stddev=0.01))
        f_b2 = tf.Variable(tf.truncated_normal([self.C], stddev=0.01))
        #------------------------------------------#
        self.pt = utils.add_layer(self.xt, f_w1, f_b1, tf.nn.leaky_relu)
        self.pt = tf.nn.l2_normalize(self.pt, dim = 1)
        self.pl = tf.slice(self.pt, [0, 0], [self.nl, -1]) # extract pl from pt
        self.pu = tf.slice(self.pt, [self.nl, 0], [self.nu, -1]) # extract pu from pt        
        #------------------------------------------#
        self.all_data = tf.concat([self.ps, self.pt], 0)
        self.f_x_logits = utils.add_layer(self.all_data, f_w2, f_b2, None)
        self.f_xs_logits = tf.slice(self.f_x_logits, [0, 0], [self.ns, -1]) # extract f_xs_logists from f_x_logists
        self.f_xl_logits = tf.slice(self.f_x_logits, [self.ns, 0], [self.nl, -1]) # extract f_xl_logists from f_x_logists
        self.f_xu_logits = tf.slice(self.f_x_logits, [self.ns + self.nl, 0], [self.nu, -1]) # extract f_xu_logists from f_x_logists
        self.pseudo_yu = tf.nn.softmax(self.f_xu_logits)
        #------------------------------------------#
        with tf.name_scope('loss_f_xl'):
            self.loss_f_xl = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.yl, logits=self.f_xl_logits))
        with tf.name_scope('loss_f_xs'):
            self.loss_f_xs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ys, logits=self.f_xs_logits))
        with tf.name_scope('loss_mmd'):
          self.loss_mmmd, self.loss_cmmd, self.loss_mmd = utils.compute_loss_mmd(self.ps, self.pl, self.pu, self.ys, self.yl, self.pseudo_yu, self.C, self.nu)
        #------------------------------------------#
        # reguralization term
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(f_w1))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(f_w2))
        self.loss_reg = tf.add_n(tf.get_collection("loss"))
        #------------------------------------------#
        with tf.name_scope('loss_total'):
            self.loss_total = self.loss_f_xl + self.beta * self.loss_f_xs + self.mu * self.loss_mmd + self.loss_reg
        # the accuracy of xl
        pred_xl = tf.nn.softmax(self.f_xl_logits)
        correct_pred_xl = tf.equal(tf.argmax(self.yl,1), tf.argmax(pred_xl,1))
        self.xl_acc = tf.reduce_mean(tf.cast(correct_pred_xl, tf.float32))
        # the accuracy of ps
        pred_xs = tf.nn.softmax(self.f_xs_logits)
        correct_pred_xs = tf.equal(tf.argmax(self.ys,1), tf.argmax(pred_xs,1))
        self.xs_acc = tf.reduce_mean(tf.cast(correct_pred_xs, tf.float32))
        # the accuracy of xu
        correct_pred_xu = tf.equal(tf.argmax(self.yu,1), tf.argmax(self.pseudo_yu,1))
        self.xu_acc = tf.reduce_mean(tf.cast(correct_pred_xu, tf.float32))
        #------------------------------------------#
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss_total)
        #self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss_total)
        #self.train_step = tf.train.MomentumOptimizer(self.lr, 0.05).minimize(self.loss_total)

        #writer = tf.summary.FileWriter("log/", self.sess.graph)
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.sess.run(init)

