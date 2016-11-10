"""
This is the module responsible for generating new recipes.

The RNN class is based HEAVILY on hunkim's work:
https://github.com/hunkim/word-rnn-tensorflow
"""

from tqdm import tqdm
import time
import myio.myio as myio
import chef_global.config as config
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from statistics.textloader import TextLoader
from six.moves import cPickle
import os
import random
from statistics.statsutils import SentenceIterator
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import gensim

rnn_size = 256
num_layers = 2
batch_size = 50
seq_length = 25
grad_clip = 5
num_epochs = 50
learning_rate = 0.002
decay_rate = 0.97
save_every = 500

def _encode_bit_vector(bit_vector):
    """
    Encodes the given vector into a compressed
    format and returns the compressed version.
    @param bit_vector: A vector of 1s and 0s, which is mostly 0s.
    @return: The compressed version of the bit vector.
    """
    # TODO: Apply Golomb's or IRL on the bit vector
    pass


def _encode_training_data(rec_table):
    """
    Encodes all of the ingredients into their compressed
    format and pairs them into lists that belong together
    according to recipes, then tuples that with the recipe
    itself.
    @param rec_table: A fully loaded RecipeTable object
                      that has recipes with ingredients
                      and text.
    @return:void, but a list of tuples of the form
            ([encoded ingredients], recipe) are written
            to disk.
    """
    training_data = []
    for recipe in tqdm(rec_table):
        encoded_ingredients = [_encode_bit_vector(ingredient)
                                for ingredient in recipe]
        data_point = (encoded_ingredients, recipe)
        training_data.append(data_point)
    myio.save_pickle(training_data, config.TRAINING_PATH)



def _generate_recipe(ingredients, rec_table, ing_table):
    """
    The main API function for this module.
    Takes a list of ingredients and a recipe table.
    Turns the list of ingredients into feature vectors,
    then compresses them into much smaller vectors and
    feeds those into the trained RNN.
    Prints the recipe that gets generated.
    @param ingredients: A list of ingredients to use in the recipe
    @param rec_table: A RecipeTable object that can be used to convert
                      the ingredients to feature vectors.
    @return: void
    """
    feature_vectors = [rec_table.ingredient_to_feature_vector(ingredient)
                            for ingredient in ingredients]

    encoded_fvs = [_encode_bit_vector(fv) for fv in feature_vectors]

    generated_recipe = __get_recipe_from_rnn(encoded_fvs, " ".join(ingredients), ing_table)

    print("Generated this recipe for you: ")
    print(str(generated_recipe))


def _train_rnn(rec_table):
    """
    Loads the training data from disk and uses it
    to train the RNN.
    @param rec_table: A fully loaded RecipeTable object
    @return: void
    """
    print("    |-> Generating word2vec model for the RNN and saving it...")
    sentences = SentenceIterator(os.path.join(config.RNN_DATA_DIR, "input.txt"))
    vec_model = gensim.models.Word2Vec(sentences, min_count=1, workers=4, iter=5)
    vec_model.save(os.path.join(config.CHECKPOINT_DIR, "word2vec.model"))

    data_loader = TextLoader(config.RNN_DATA_DIR, batch_size, seq_length)
    vocab_size = data_loader.vocab_size

    with open(os.path.join(config.CHECKPOINT_DIR, "words_vocab.pkl"), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)

    model = MyRNN(rnn_size, num_layers, batch_size, seq_length, vocab_size, grad_clip)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        ####### CHANGE THIS TO LOAD A CHECKPOINTED MODEL #########
        restore = False
        ##########################################################
        if restore:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            checkpoint = tf.train.get_checkpoint_state(config.CHECKPOINT_DIR)
            with open(os.path.join(config.CHECKPOINT_DIR, "words_vocab.pkl"), 'rb') as f:
                saved_words, saved_vocab = cPickle.load(f)

        for e in range(num_epochs):
            sess.run(tf.assign(model.lr, learning_rate * (decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            print("Training RNN: epoch " + str(e) + " of " + str(num_epochs))
            for b in tqdm(range(data_loader.num_batches)):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op],
                                                        feed)
                end = time.time()
                if (e * data_loader.num_batches + b) % save_every == 0 \
                        or (e == num_epochs - 1 and b == data_loader.num_batches - 1):
                    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("Model saved to " + str(checkpoint_path))


def __get_recipe_from_rnn(encoded_feature_vectors, ingredients, ing_table):
    """
    Feeds the given bit vectors into the neural network
    and has it generate a recipe.
    @param encoded_feature_vectors: A list of encoded ingredients
                                    to use in the recipe.
    @param ingredients: The ingredients
    @return: The generated recipe
    """
    training_data = myio.load_pickle(config.TRAINING_PATH)

    with open(os.path.join(config.CHECKPOINT_DIR, "words_vocab.pkl"), 'rb') as f:
        words, vocab = cPickle.load(f)

    data_loader = TextLoader(config.RNN_DATA_DIR, batch_size, seq_length)
    vocab_size = data_loader.vocab_size
    model = MyRNN(rnn_size, num_layers, batch_size, seq_length, vocab_size, grad_clip, infer=True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(config.CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            vec_model = gensim.models.Word2Vec.load(os.path.join(\
                                config.CHECKPOINT_DIR, "word2vec.model"))
            return model.sample(sess, words, vocab, vec_model, ingredients, ing_table)#prime="apple")#ingredients)
        else:
            print("Could not locate trained model in " + str(config.CHECKPOINT_DIR))
            return None



class MyRNN:
    """
    RNN class.

    Based heavily on hunkim's:
    https://github.com/hunkim/word-rnn-tensorflow
    """
    def __init__(self, rnn_size, num_layers, batch_size, seq_length, vocab_size, grad_clip,\
                         infer=False):
        """
        Constructor for an RNN using LSTMs.
        @param rnn_size: The size of the RNN
        @param num_layers: The number of layers for the RNN to have
        @param batch_size: The batch size to train with
        @param seq_length: The length of the sequences to use in training
        @param vocab_size: The size of the vocab
        @param grad_clip: The point at which to clip the gradient in the gradient descent
        @param infer:
        """
        if infer:
            batch_size = 1
            seq_length = 1

        cell_fn = rnn_cell.BasicLSTMCell
        cell = cell_fn(rnn_size)
        self.cell = cell = rnn_cell.MultiRNNCell([cell] * num_layers)

        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope("rnnlm"):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size])
            with(tf.device("/cpu:0")):
                embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
                inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(\
                                                    embedding, self.input_data))
                inputs = [tf.squeeze(inp, [1]) for inp in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        loop_func = loop if infer else None
        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state,\
                                        cell, loop_function=loop_func, scope="rnnlm")
        output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],\
                            [tf.reshape(self.targets, [-1])],\
                            [tf.ones([batch_size * seq_length])], vocab_size)
        self.cost = tf.reduce_sum(loss) / batch_size / seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, session, words, vocab, word2vec_model, ingredients, table,\
                         num=0, prime="first"):
        """
        Samples the model's results by feeding it words, vocabulary, and asking
        for a number of words to get out.
        @param session:
        @param words:
        @param vocab:
        @param table: An ingredients table
        @param num:
        @param prime:
        @return:
        """
        state = session.run(self.cell.zero_state(1, tf.float32))
        if not len(prime) or prime == " ":
            prime = random.choice(list(vocab.keys()))

        print("Prime that was chosen for this sample: ", str(prime))
        for word in prime.split()[:-1]:
            print(word)
            x = np.zeros((1, 1))
            x[0, 0] = vocab.get(word, 0)
            feed = {self.input_data: x, self.initial_state: state}
            [state] = session.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return (int(np.searchsorted(t, np.random.rand(1) * s)))

        ret = prime
        word = prime.split()[-1]

        def choose_next_word(state):
            x = np.zeros((1, 1))
            x[0, 0] = vocab.get(word, 0)
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = session.run([self.probs, self.final_state], feed)
            p = probs[0]

            sample = weighted_pick(p)

            # TODO: This is a hack for now until I figure out why sample can
            # sometimes be bigger than len(words)
            # May have been solved - try without the hack TODO
            while sample > len(words):
                sample = weighted_pick(p)

            pred = words[sample]
            return pred

        def is_too_similar(word):
            threshold = 0.25
            for ingredient in ingredients.split(" "):
                try:
                    similarity = word2vec_model.similarity(ingredient, word)
                    if similarity > threshold and len(ingredient) > 1:
                        return True
                except KeyError:
                    return False
            return False

        def word_is_ingredient(word):
            return table.has(word) and word not in ingredients.split(" ")

        def get_best_match(word):
            return word2vec_model.most_similar(positive=[word])[0]

        def get_most_similar(word):
            if len(ingredients) > 0:
                best_match = ""
                best_sim = -100;
                for ingredient in ingredients.split(" "):
                    try:
                        similarity = word2vec_model.similarity(word, ingredient)
                        if similarity > best_sim and len(ingredient) > 1:
                            best_match = ingredient
                            best_sim = similarity
                    except KeyError:
                        pass
                return best_match
            else:
                return word

        # TODO
        num = 200
        if num is 0:
            n = 0
            while True:
                word = choose_next_word(state)
                if word.lower() == config.NEW_RECIPE_LINE.lower():
                    break
                else:
                    n += 1
                    if is_too_similar(word):
                        word = get_most_similar(word)
                    elif word_is_ingredient(word):
                        word = get_most_similar(word)
                    ret += " " + word
        else:
            for n in range(num):
                word = choose_next_word(state)
                if is_too_similar(word):
                    word = get_most_similar(word)
                elif word_is_ingredient(word):
                    word = get_most_similar(word)
                ret += " " + word
        return ret










































