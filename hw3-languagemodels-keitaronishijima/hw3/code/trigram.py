from functools import reduce
import numpy as np
import tensorflow as tf
from preprocess import get_data
from tensorflow.keras import Model
import os

# ensures that we run only on cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()
        
        def make_vars(*dims, initializer=tf.random.normal): 
            return tf.Variable(tf.random.normal(dims, stddev=.1, dtype=tf.float32))
        # TODO: initialize emnbedding_size, batch_size, and any other hyperparameters
        self.vocab_size = vocab_size
        self.embedding_size = 100  # TODO
        self.batch_size = 1000  # TODO
        self.epoch = 5
        self.learning_rate = 0.001
        # TODO: initialize embeddings and forward pass weights (weights, biases)
        self.emb = tf.keras.layers.Embedding(vocab_size, self.embedding_size)
        
        self.E = make_vars(vocab_size, self.embedding_size)   ## TODO
        self.W = make_vars(self.embedding_size *2, vocab_size)
        self.b = make_vars(vocab_size)
        
    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network
        (i.e. tf.nn.embedding_lookup)

        :param inputs: word ids of shape (batch_size, 2)
        :return: probabilities: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """

        # TODO: Fill in
        t1 =  tf.nn.embedding_lookup(self.E, inputs[:, 0])
        t2 = tf.nn.embedding_lookup(self.E, inputs[:, 1])
        embed = tf.concat([t1,t2], 1)
        return tf.nn.softmax(tf.nn.relu(tf.matmul(embed , self.W) + self.b))

    def loss_function(self, probabilities, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probabilities: a matrix of shape (batch_size, vocab_size)
        :return: the average loss of the model as a tensor of size 1
        """
        # TODO: Fill in
        # We recommend using tf.keras.losses.sparse_categorical_crossentropy
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities))


def train(model, train_input, train_labels):
    """
    Runs through one epoch - all training examples.
    Remember to shuffle your inputs and labels - ensure that they are shuffled
    in the same order. Also you should batch your input and labels here.

    :param model: the initilized model to use for forward and backward pass
    :param train_input: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_input: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate= model.learning_rate)
    # TODO Fill in
    idx_list = list(range(0, len(train_labels)))
    idx_list = tf.random.shuffle(idx_list)
    train_labels = tf.gather(train_labels, idx_list)
    train_input = tf.gather(train_input, idx_list)
    for i in range(len(train_labels) // model.batch_size):
        X_batch = train_input[i * model.batch_size: (i+1) * model.batch_size]
        Y_batch = train_labels[i * model.batch_size: (i+1) * model.batch_size]
        with tf.GradientTape() as tape:
            logits = model.call(X_batch)
            loss = model.loss_function(logits, Y_batch)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_input, test_labels):
    """
    Runs through all test examples. Test input should be batched here.

    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)
    :returns: perplexity of the test set
    """

    # TODO: Fill in
    # NOTE: Ensure a correct perplexity formula (different from raw loss)
    perplexity = 0
    count = 0
    for i in range(len(test_labels) // model.batch_size):
        X_batch = tf.convert_to_tensor(test_input[i * model.batch_size: (i+1) * model.batch_size])
        Y_batch = tf.convert_to_tensor(test_labels[i * model.batch_size: (i+1) * model.batch_size])
        logits = model.call(X_batch)
        average_loss = model.loss_function(logits, Y_batch)
        perplexity += np.e ** average_loss
        count += 1
    return perplexity / count


def generate_sentence(word1, word2, length, vocab, model):
    """
    Given initial 2 words, print out predicted sentence of targeted length.

    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model

    """

    # NOTE: This is a deterministic, argmax sentence generation

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    output_string = np.zeros((1, length), dtype=np.int)
    output_string[:, :2] = vocab[word1], vocab[word2]

    for end in range(2, length):
        start = end - 2
        output_string[:, end] = np.argmax(model(output_string[:, start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]

    print(" ".join(text))


def main():
    # TODO: Pre-process and vectorize the data using get_data from preprocess
    train_data, test_data, word2id = get_data('data/train.txt', 'data/test.txt')
    # TO-DO:  Separate your train and test data into inputs and labels
    train_inputs = []
    train_labels = []
    for i in range(len(train_data)-2): 
        first = train_data[i]
        second = train_data[i+1]
        third = train_data[i+2]
        # check if end of sentence or the input contains endin
        if first == 34 or second == 34 or third == 34:
            continue
        train_inputs.append([first, second])
        train_labels.append(third)
    test_inputs = []
    test_labels = []
    for i in range(len(test_data)-2):
        first = test_data[i]
        second = test_data[i+1]
        third = test_data[i+2]
        # check if end of sentence or the input contains endin
        if first == 34 or second == 34 or third == 34:
            continue
        test_inputs.append([first, second])
        test_labels.append(third)
    # TODO: initialize model
    model = Model(vocab_size = len(word2id))
    # TODO: Set-up the training step
    for i in range(model.epoch):
        train(model,train_inputs, train_labels)
    # TODO: Set up the testing steps
    perplexity = test(model, test_inputs, test_labels)
    # Print out perplexity
    print(perplexity)
    # BONUS: Try printing out sentences with different starting words
    generate_sentence("theory", "of", 5, word2id, model)


if __name__ == "__main__":
    main()
