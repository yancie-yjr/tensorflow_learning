from datetime import datetime
import math
import time
import tensoeflow as tf

###
batch_size=32
num_batchs=100

### 显示网络每一层结构的函数，展示每一个卷积层活池化层输出的tensor的尺寸
def print_activations(t)
    print(t.op.name,'',t.get_shape().as_list())
  
### 网络结构
# 使用TensorFlow中的name_scope,通过with tf.name_scope('conv1') as scope可以将scope内生成的
# Variable自动命名为conv1/xxx
def inference(images):
    parameters=[]
    #第一层
    with tf.name_scope('conv1') as scope:
        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape[64],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope)
        print_activations(conv1)
        parameters+=[kernel,biases]

    lrn1=tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
    pool1=tf.nn.max_pool(lrn,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
    
    #第二层
    with tf.name_scope('conv2') as scope:
        kernel=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape[192],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias,name=scope)
        print_activations(conv2)
        parameters+=[kernel,biases]

    lrn2=tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
    pool2=tf.nn.max_pool(lrn,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
    
    #第三层
    with tf.name_scope('conv3') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape[384],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv3=tf.nn.relu(bias,name=scope)
        print_activations(conv3)
        parameters+=[kernel,biases]

    #第四层
    with tf.name_scope('conv4') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape[256],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv4=tf.nn.relu(bias,name=scope)
        print_activations(conv3)
        parameters+=[kernel,biases]

    #第五层
    with tf.name_scope('conv5') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape[256],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv5=tf.nn.relu(bias,name=scope)
        print_activations(conv3)
        parameters+=[kernel,biases]
    
    pool5=tf.nn.max_pool(lrn,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
    
    return pool5,parameters
  


















     
