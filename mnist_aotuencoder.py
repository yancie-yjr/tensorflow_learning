import numpy as np
import sklearn.preprocessing as pre
import tensorflow as tf
from tensorflow.example.tutorials.mnist import input_data

### 一种合理的初始化方式
def xaviar_init(fan_in,fan_out,constant=1)         #fan_in,fan_out为输入和输出节点的个数
    low=-constant*np.sqrt(6.0/fan_in+fan_out)
    high=-constant*np.sqrt(6.0/fan_in+fan_out)
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

### 定义一个去噪自编码的class
# n_input输入变数,n_hidden隐含层节点数,transfer_function=tf.nn.softplus隐含层激活函数,
# optimizer=tf.train.AdamOptimizer()优化器,scale=0.1高斯噪声系数
class AdditiveGaussianNoideAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(),scale=0.1)
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function
        self.training_scale=scale
        network_weights=self._initialize_weights()
        self.weights=network_weights

# 定义网络结构
        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden=self.transfer(tf.add(tf.matual(self.x+scale*tf.random_normal((n_input,)),
                           self.weights['W1']),self.weights['B1']))
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['W2']),
                           self.weights['b2'])










