"""
Module to hold the encoder model.
"""

from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import matplotlib
try:
    if os.environ["SSH_CONNECTION"]:
        matplotlib.use("Pdf")
except KeyError:
    pass
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Network Parameters
n_hidden_1 = 512#256 # 1st layer num features
n_hidden_2 = 256#128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    multiplied = tf.matmul(x, weights["encoder_h1"])
    added = tf.add(multiplied, biases["encoder_b1"])
    layer_1 = tf.nn.sigmoid(added)

    # Decoder Hidden layer with sigmoid activation #2
    multiplied = tf.matmul(layer_1, weights["encoder_h2"])
    added = tf.add(multiplied, biases["encoder_b2"])
    layer_2 = tf.nn.sigmoid(added)

    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    multiplied = tf.matmul(x, weights["decoder_h1"])
    added = tf.add(multiplied, biases["decoder_b1"])
    layer_1 = tf.nn.sigmoid(added)

    # Decoder Hidden layer with sigmoid activation #2
    multiplied = tf.matmul(layer_1, weights["decoder_h2"])
    added = tf.add(multiplied, biases["decoder_b2"])
    layer_2 = tf.nn.sigmoid(added)

    return layer_2

# Construct model
print("Constructing the model...")
print("    |-> Constructing the encoder model...")
encoder_op = encoder(X)
print("    |-> Constructing the decoder model...")
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
learning_rate = 0.01
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
training_epochs = 20
batch_size = 256
print("Launch the tensorflow session...")
with tf.Session() as sess:
    print("    |-> Run the session...")
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    print("TOTAL BATCH: " + str(total_batch))
    exit(0)
    # Training cycle
    print("    |-> Run the training...")
    for epoch in tqdm(range(training_epochs)):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            print("Batches are done: " + str(batch_xs))
            print("Batch ys: " + str(batch_ys))
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

    print("    |-> Optimization finished.")

    # Applying encode and decode over test set
    examples_to_show = 10
    encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
