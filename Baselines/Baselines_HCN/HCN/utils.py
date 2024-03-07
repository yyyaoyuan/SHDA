import tensorflow as tf
import numpy as np
#=========================================================================#
#-----------------------------------------#
# define the build_s network
def build_s(x, weights, biases, activation_function):
    layer_1 = tf.add(tf.matmul(x, weights['s_w1']), biases['s_b1'])
    if activation_function:
       layer_1 = activation_function(layer_1)
    #---------------------#
    layer_1 = tf.nn.l2_normalize(layer_1, dim = 1)
    return layer_1
#-----------------------------------------#
# define the build_t network
def build_t(x, weights, biases, activation_function):
    layer_1 = tf.add(tf.matmul(x, weights['t_w1']), biases['t_b1'])
    if activation_function:
       layer_1 = activation_function(layer_1)
    #---------------------#
    layer_1 = tf.nn.l2_normalize(layer_1, dim = 1)
    return layer_1
#-----------------------------------------#
# define full-connect layer
def add_layer(x, weights, biases, activation_function):
    Wx_plus_b = tf.add(tf.matmul(x, weights), biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
#=========================================================================#
# computer Dis_MMD 
def calculate_dis_mmd(xs, ys, class_number):
    xs_label = tf.argmax(ys, 1)
    dis_mmd = tf.constant(0., tf.float32)
    for i in range(class_number - 1):
        index_xs_i = tf.cast(tf.equal(xs_label, i), tf.int32)
        xs_i = tf.dynamic_partition(xs, index_xs_i, 2)[1]
        mean_xs_i = tf.reduce_mean(xs_i, 0)
        for j in range(i, class_number):
            index_xs_j = tf.cast(tf.equal(xs_label, j), tf.int32)
            xs_j = tf.dynamic_partition(xs, index_xs_j, 2)[1]
            mean_xs_j = tf.reduce_mean(xs_j, 0)
            dis_mmd += tf.reduce_sum(tf.square(mean_xs_i - mean_xs_j))
    dis_mmd = dis_mmd * 1. / (class_number * (class_number - 1) / 2.)
    return dis_mmd
#--------------------------------------------#
# computer Dis_MMD 
def calculate_dis_mmd_1(xs, ys, class_number):
    xs_label = tf.argmax(ys, 1)
    dis_mmd = tf.constant(0., tf.float32)
    for i in range(class_number):
        index_xs_i = tf.cast(tf.equal(xs_label, i), tf.int32)
        xs_i = tf.dynamic_partition(xs, index_xs_i, 2)[1]
        mean_xs_i = tf.reduce_mean(xs_i, 0)
        for j in range(class_number):
            index_xs_j = tf.cast(tf.equal(xs_label, j), tf.int32)
            xs_j = tf.dynamic_partition(xs, index_xs_j, 2)[1]
            mean_xs_j = tf.reduce_mean(xs_j, 0)
            dis_mmd += tf.reduce_sum(tf.square(mean_xs_i - mean_xs_j))
    dis_mmd = dis_mmd * 1. / (class_number * (class_number - 1))
    return dis_mmd
#=========================================================================#
# computer MMD loss
def calculate_loss_mmd(xs, xl, ys, yl, class_number):
    d = tf.shape(xl)[1]
    mean_xs = tf.reduce_mean(xs, 0)
    mean_xl = tf.reduce_mean(xl, 0)
    margin_loss = tf.reduce_sum(tf.square(mean_xs - mean_xl))
    #---------------------------------------#
    xs_label = tf.argmax(ys,1)
    xl_label = tf.argmax(yl,1)
    conditional_loss = tf.constant(0., tf.float32)
    for k in range(class_number):
        index_xs_k = tf.cast(tf.equal(xs_label,k), tf.int32)
        index_xl_k = tf.cast(tf.equal(xl_label,k), tf.int32)
        xs_k = tf.dynamic_partition(xs,index_xs_k,2)[1]
        xl_k = tf.dynamic_partition(xl,index_xl_k,2)[1]
        #---------------------------------#
        mean_xs_k = tf.reduce_mean(xs_k, 0)
        mean_xl_k = tf.reduce_mean(xl_k, 0)
        #---------------------------------#
        conditional_loss += tf.reduce_sum(tf.square(mean_xs_k - mean_xl_k))
    mmd_loss = (margin_loss + conditional_loss) / (class_number + 1.)
    return margin_loss, conditional_loss, mmd_loss
#=========================================================================#
