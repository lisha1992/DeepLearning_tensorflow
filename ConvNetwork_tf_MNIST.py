# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:55:42 2016
Convolutional Network for MNIST


@author: ceciliaLee
"""

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes=10 # 0-9 classes
dropout=0.75 # Dropout, probability to keep units


# Tensorflow graph input
x = tf.placeholder(tf.float32, [None, 784]) # mnist image 
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 classes
keep_prob=tf.placeholder(tf.float32) # Dropout, probability to keep units

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # w 表示一个卷积核
    """
    Conv 2D wrapper with bias and ReLU activation 

    卷积层是卷积核在上一级输入层上通过逐一滑动窗口计算而得，
    卷积核中的每一个参数都相当于传统神经网络中的权值参数，
    与对应的局部像素相连接，将卷积核的各个参数与对应的局部像素值相乘之和，
    （通常还要再加上一个偏置参数），得到卷积层上的结果

    """
    x=tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME') #[batch, in_height, in_width, in_channels]
    # Computes a 2-D convolution given 4-D `input` and `filter` tensors
    # strides: 使用滑动步长为1的窗口
    # padding='SAME'表示通过填充0，使得输入和输出的形状一致
    x=tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    
def maxpool2d(x, k=2):
    # Maxpool 2D Wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME') 
    #ksize 中的2,2表示池化的时候的池化区域大小
    
    
## Create model
def conv_net(x, weights, biases, dropout):


    # Reshape input picture
    x=tf.reshape(x, shape=[-1, 28, 28, 1]) 
    # 为了使得图片与计算层匹配，我们首先reshape输入图像x为4维的tensor，第2、3维对应图片的宽和高，最后一维对应颜色通道的数目
    
    # Convolutional layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max pooling (doen sampling)
    conv1 = maxpool2d(conv1, k=2)
    
    # Convolutional layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max pooling (doen sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1=tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1=tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1=tf.nn.relu(fc1)
    # Apply Dropout for avoiding overfitting
    fc1=tf.nn.dropout(fc1, dropout)
    
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
    
# Store layers weights and bias
weights={
    # 5x5 conv, 1 input, 32 outputs 卷积在每个5x5的patch中算出32个特征
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), #前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), # stack network layers
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])), # down sampled to 7*7
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes])) # fully-connected layer with 1024 units
    }
biases={
    'bc1': tf.Variable(tf.random_normal([32])),  # one bias for each output channel
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
# Construct model
pred=conv_net(x, weights, biases, keep_prob)

# Define loss function and optimization
cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

## Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

## Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        # Run optimization operation (back-propagation)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
        step += 1
   
   
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256],keep_prob: 1.}))

    
