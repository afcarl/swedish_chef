"""
Module to hold the encoder model.
"""

from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import random
import matplotlib
try:
    if os.environ["SSH_CONNECTION"]:
        matplotlib.use("Pdf")
except KeyError:
    pass
import matplotlib.pyplot as plt


random.seed(456)

# Model parameters
fv_length = 200#fixed
n_hidden_1 = 64#
n_hidden_2 = 32#
n_input = fv_length
training_epochs = 100
batch_size = 256#fixed

# 200, 100, 50, 10, 256 -> 55%
# 200, 128, 50, 10, 256 -> 57%
# 200, 100, 80, 10, 256 -> 57%
# 200, 100, 100, 10, 256 -> 55%
# 200, 64, 32, 10, 256 -> 56%

# 200, 64, 32, 40, 256 -> 63%
# 200, 64, 32, 100, 256 -> 57%

def get_random_batch(batch_size):
    """
    Gets a random batch of test data using the random package, which
    has already been seeded.
    @param batch_size: The number of data points to get
    @return: A tuple of the form (data, labels)
    """
    batch = []
    for j in range(batch_size):
        data_point = [random.randint(0, 1) for i in range(fv_length)]
        batch.append(data_point)

    batch = np.matrix(batch)
    return (batch, batch)

get_random_batch(3)
weights = {
    "encoder_h1": tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    "encoder_h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    "decoder_h1": tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    "decoder_h2": tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

biases = {
    "encoder_b1": tf.Variable(tf.random_normal([n_hidden_1])),
    "encoder_b2": tf.Variable(tf.random_normal([n_hidden_2])),
    "decoder_b1": tf.Variable(tf.random_normal([n_hidden_1])),
    "decoder_b2": tf.Variable(tf.random_normal([n_input])),
}

def encoder(x):
    """
    Builds the encoder network.
    @param x:
    @return:
    """
    # Encoder Hidden layer with sigmoid activation #1
    multiplied = tf.matmul(x, weights["encoder_h1"])
    added = tf.add(multiplied, biases["encoder_b1"])
    layer_1 = tf.nn.sigmoid(added)

    # Decoder Hidden layer with sigmoid activation #2
    multiplied = tf.matmul(layer_1, weights["encoder_h2"])
    added = tf.add(multiplied, biases["encoder_b2"])
    layer_2 = tf.nn.sigmoid(added)

    return layer_2


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
X = tf.placeholder("float", [None, n_input])
encoder_op = encoder(X)

print("    |-> Constructing the decoder model...")
decoder_op = decoder(encoder_op)

y_predicted = decoder_op
y_labels = X

# Define loss and optimizer, minimize the squared error
learning_rate = 0.01
cost = tf.reduce_mean(tf.pow(y_labels - y_predicted, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
print("Launch the tensorflow session...")
with tf.Session() as sess:
    print("    |-> Run the session...")
    sess.run(init)
    # TODO: Compute this
    total_batch = 214
    #total_batch = int(mnist.train.num_examples/batch_size)
    print("    |-> Run the training...")
    for epoch in range(training_epochs):
        print("    |-> EPOCH " + str(epoch + 1) + " of " + str(training_epochs))
        for i in tqdm(range(total_batch)):
            # TODO: What is this function?
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs, batch_ys = get_random_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
    print("    |-> Optimization finished.")

    # Applying encode and decode over test set
    test_data = get_random_batch(1)
    # TODO: Maybe like this?
    encode_decode = sess.run(y_predicted, feed_dict={X: test_data[0]})
    #encode_decode = sess.run(y_predicted, feed_dict={X: mnist.test.images[:examples_to_show]})

    encode = test_data[0].getA1()
    decode = encode_decode[0]

    total = 0
    for i in range(len(encode)):
        encode[i] = int(encode[i] + 0.5)
        decode[i] = int(decode[i] + 0.5)
        if encode[i] == decode[i]:
            #print("Values: " + str(encode[i]) + " and " + str(decode[i]) + " matched.")
            total += 1
            #print("Total correct " + str(total) + " out of " + str(i + 1))
        else:
            #print("Values: " + str(encode[i]) + " and " + str(decode[i]) + " did not match.")
            #print("Total correct " + str(total) + " out of " + str(i + 1))
            pass

    print("Encoded: " + str(encode))
    print("Decoded: " + str(decode))

    accuracy = total / len(encode)
    print("Accuracy: "  + str(accuracy))







