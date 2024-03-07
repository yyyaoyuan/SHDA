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
