import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
#=========================================================================#
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
#-----------------------------------------#
# computer MMD loss
def compute_loss_mmd(ps, pl, pu, ys, yl, pseudo_yu, C, nu):
    xt = tf.concat([pl, pu], 0)
    d = tf.shape(xt)[1]
    mean_ps = tf.reduce_mean(ps, 0)
    mean_xt = tf.reduce_mean(xt, 0)
    loss_mmmd = tf.reduce_sum(tf.square(mean_ps - mean_xt))
    #---------------------------------------#
    ps_label = tf.argmax(ys,1)
    pl_label = tf.argmax(yl,1)
    loss_cmmd = tf.constant(0., tf.float32)
    for k in range(C):
        index_ps_k = tf.cast(tf.equal(ps_label,k), tf.int32)
        index_pl_k = tf.cast(tf.equal(pl_label,k), tf.int32)
        ps_k = tf.dynamic_partition(ps,index_ps_k,2)[1]
        pl_k = tf.dynamic_partition(pl,index_pl_k,2)[1]
        #---------------------------------#
        mean_ps_k = tf.reduce_mean(ps_k, 0)
        #---------------------------------#
        sum_pl_k = tf.reduce_sum(pl_k, 0)
        weight = tf.reshape(pseudo_yu[:,k], [-1,1])
        weight_pu_k = tf.multiply(pu, tf.tile(weight, [1,d]))
        sum_pu_k = tf.reshape(tf.reduce_sum(weight_pu_k, 0), [1,-1])
        nl_k = tf.cast(tf.shape(pl_k)[0], tf.float32)
        nu_k = tf.reduce_sum(weight)
        mean_xt_k = (sum_pl_k + sum_pu_k) / (nl_k + nu_k)
        #---------------------------------#
        loss_cmmd += tf.reduce_sum(tf.square(mean_ps_k - mean_xt_k))
    loss_mmd = loss_mmmd + loss_cmmd
    loss_mmd = loss_mmd * 1.0 / (C + 1.0)
    return loss_mmmd, loss_cmmd, loss_mmd
#-----------------------------------------#
