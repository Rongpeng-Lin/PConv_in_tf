########   ops
import math
import tensorflow as tf
def conv(name,x,ker_size,outs,s,cur_bin_m,mask_convd,new_bin_m,if_bn=True):
    ker = int(math.sqrt(ker_size))
    x_shape = [i.value for i in x.get_shape()]
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [ker,ker,x_shape[-1],outs],
                            tf.float32,
                            tf.initializers.truncated_normal(stddev=0.02))
        b = tf.get_variable('b',
                            [outs],
                            tf.float32,
                            tf.initializers.constant(0.))
        return (tf.nn.conv2d(x*cur_bin_m,w,[1,s,s,1],"SAME")*mask_convd+b)*new_bin_m
     
def ins_norm(name,x,new_bin_m):
    with tf.variable_scope(name):
        x_ins = tf.contrib.layers.instance_norm(x)
        return x_ins*new_bin_m
        
def relu(name,x):
    with tf.variable_scope(name):
        return tf.nn.relu(x)
        
def tanh(name,x):
    with tf.variable_scope(name):
        return tf.nn.tanh(x)

def conv_up(name, in1, in1_mask, in2, in2_mask, maskconvd, masknewbin, ker_size, outs, s):
    ker = int(math.sqrt(ker_size))
    in_ch = [i.value for i in in1.get_shape()][-1] + [i.value for i in in2.get_shape()][-1]
    hi = [i.value for i in in1.get_shape()][1]
    with tf.variable_scope(name):
        in_up = tf.image.resize_images(in1,
                                       [int(2*hi),int(2*hi)], 
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        w = tf.get_variable('w',
                            [ker,ker,in_ch,outs],
                            tf.float32,
                            tf.initializers.truncated_normal(stddev=0.02))
        b = tf.get_variable('b',
                            [outs],
                            tf.float32,tf.initializers.constant(0.))
        
        return (tf.nn.conv2d(tf.concat([in_up*in1_mask,in2*in2_mask],3),w,[1,s,s,1],"SAME")*maskconvd+b)*masknewbin
