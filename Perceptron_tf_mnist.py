# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:51:35 2016
MNIST-Perceptron

@author: ceciliaLee
"""

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1


# Network parameters
n_hidden_1=256 # 1st layer number of features
n_hidden_2=256 # 2nd layer number of features
n_input=784 # MNIST data input (img shape: 28*28)
n_classes=10 # 0-9 class

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Creat model
def multilayer_perceptron(x, weights, biases):
    # Hidden layers with ReLU activation - 1st layer
    layer_1=tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1=tf.nn.relu(layer_1)
    # 2nd layer
    layer_2=tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2=tf.nn.relu(layer_2)    
    
    # Output with linear layer
    output_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    
    return output_layer
    
# Store layers weight and bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
biases={
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }

# Construct model

#pred = multilayer_perceptron(x, weights, biases)
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init=tf.initialize_all_variables()

## Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


