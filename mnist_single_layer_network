from tensorflow.examples.tutorials.minist import input_data
import tensorflow as tf

# 下载MNIST数据集，标签用one_hot
# 训练集55000，验证集5000，测试集10000,照片大小28*28
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

# placeholder是输入数据的地方，参数[数据类型，shape]，第二个参数[None,784]，None代表batch还没有定
sess=tf.InteractiveSession()

### 定义网络
### 1.定义一个占位符用于输入数据
x=tf.placeholder(tf.float32,[None,784])

### 2.学习参数，用th.Variable建立
W=th.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

### 定义预测层
### 3.预测的概率值分布
y=tf.nn.softmax(tf.matmul(x,W)+b)

### 4.定义loss function
# y_标签的输入入口
y_=tf.placeholder（tf.float32,[None,10]）
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices[1]))

### 5.设置训练节点
train_step=tf.train.GredientDesentOptimizer(0.5).minimize(cross_entropy)

### 6.全局初始化
tf.global_bariables_initializer().run()

### 7.训练
for i in range(1000)
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})

### 8.定义预测准确率
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 计算测试集的准确率
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))








