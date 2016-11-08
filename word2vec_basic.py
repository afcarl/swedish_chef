# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# Modified and adapted to use in the Swedish Chef project by Max Strange on
# 11/06/2016.
#
#
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import string
import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import myio.myio as myio
import chef_global.config as config



class WordGenerator:
    """
    Generator of words from the config.RECIPE_FILE_SINGLE_PATH.
    """
    def __init__(self):
        pass

    def __iter__(self):
        with open(config.RECIPE_FILE_SINGLE_PATH) as f:
            for line in f:
                for word in line.split(" "):
                    yield word.lower().strip().strip(string.punctuation.replace("_",""))


def build_dataset(words):
    """
    Creates a dataset to use.
    @param words: A list of words: ["the", "quick", etc.]
    @return: A tuple of the form: (data, count, dictionary, reverse_dictionary)
             where:
                data is a list token IDs
                count is a list of lists of the form [[token, number of times token occurs], etc]
                dictionary is a dict of the form {token : token ID}
                reverse dictionary is a dict of the form {token ID : token}
    """
    # Count is a list of each token along with the number of times it occurred
    # Initialize it with the uncommon token
    count = [['UNK', -1]]

    # Now populate it using python's Counter object
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    # Dictionary is a dict of token to unique ID
    dictionary = {}
    for i, item in enumerate(count):
        word = item[0]
        dictionary[word] = i

    # Data is a list of token IDs
    data = []
    unk_count = 0
    for word in words:
        index = dictionary[word] if word in dictionary else 0
        if index is 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    """
    Generates a batch of training data.
    @param batch_size: The size of the batch
    @param num_skips: The number of skips in the skip gram
    @param skip_window: The skip window for the skip gram
    @return: A tuple of the form (training batch, labels)
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                        textcoords='offset points', ha='right',
                            va='bottom')

        plt.savefig(filename)



# MAIN SCRIPT #

# Step 1: Get the words that you want to use
word_generator = WordGenerator()
words = [word for word in word_generator]
print("Number of words: ", len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.

# Step 3: Generate a training batch to train the skip-gram model
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

# Step 4: Build and train a skip-gram model.
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
                  tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                         num_sampled, vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.initialize_all_variables()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    print("Initializing tensorflow...")
    init.run()

    print("Training the neural network...")
    average_loss = 0
    for step in tqdm(xrange(num_steps)):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
    print("Generating the final embeddings...")
    final_embeddings = normalized_embeddings.eval()

    print("Saving the embeddings to " + str(config.EMBEDDINGS_PATH))
    myio.save_pickle(final_embeddings, config.EMBEDDINGS_PATH)


