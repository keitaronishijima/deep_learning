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

        # TODO: initialize embedding_size, batch_size, and any other hyperparameters
        def make_vars(*dims, initializer=tf.random.normal): 
            return tf.Variable(tf.random.normal(dims, stddev=.1, dtype=tf.float32))
        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 100  # TODO
        self.batch_size = 100  # TODO
        self.epoch = 3
        self.learning_rate = 0.01
        self.rnn_size = 512
        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN
        self.embedding = tf.keras.layers.Embedding(vocab_size, self.embedding_size)
        self.rnn_layer = tf.keras.layers.LSTM(self.rnn_size, return_sequences = True, return_state = True)
        self.forward_layer = tf.keras.layers.Dense(self.vocab_size, activation = 'softmax')

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network
        (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state
        (NOTE 1: If you use an LSTM, the final_state will be the last two RNN outputs,
        NOTE 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU
        """

        # TODO: Fill in
        # if initial_state == None:
        #     initial_state = tf.zeros((self.batch_size, self.window_size))
        embed = self.embedding(inputs)
        sq, st1, st2 = self.rnn_layer(embed, initial_state = initial_state)
        out = self.forward_layer(sq)
        return out, (st1, st2)

    def loss(self, probabilities, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probabilities: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the average loss of the model as a tensor of size 1
        """

        # TODO: Fill in
        # We recommend using tf.keras.losses.sparse_categorical_crossentropy
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples (remember to batch!)
    Here you will also want to reshape your inputs and labels so that they match
    the inputs and labels shapes passed in the call and loss functions respectively.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    window = 20
    inputs = []
    labels = []
    count = 0
    for i in range(len(train_inputs) // window):
        inputs.append(train_inputs[i * window :i*window++window])
        count += 1
    for i in range(len(train_labels) // window):
        labels.append(train_labels[i * window:i*window+window])
    train_inputs = inputs
    train_labels = labels
    # TODO: Fill in
    optimizer = tf.keras.optimizers.Adam(learning_rate= model.learning_rate)
    # TODO Fill in
    idx_list = list(range(0, len(train_labels)))
    idx_list = tf.random.shuffle(idx_list)
    train_labels = tf.gather(train_labels, idx_list)
    train_inputs = tf.gather(train_inputs, idx_list)
    rep = len(train_inputs)//model.batch_size
    for i in range(rep):
        X_batch = train_inputs[i * model.batch_size: (i+1) * model.batch_size]
        Y_batch = train_labels[i * model.batch_size: (i+1) * model.batch_size]
        with tf.GradientTape() as tape:
            logits, final = model.call(X_batch, None)
            loss = model.loss(logits, Y_batch)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples (remember to batch!)
    Here you will also want to reshape your inputs and labels so that they match
    the inputs and labels shapes passed in the call and loss functions respectively.

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    window = 20
    inputs = []
    labels = []
    for i in range(len(test_inputs) // window):
        inputs.append(test_inputs[i * window:i*window+window])
    for i in range(len(test_labels) - window):
        labels.append(test_labels[i * window :i*window+window])
    test_inputs = inputs
    test_labels = labels
    
    # TODO: Fill in
    # NOTE: Ensure a correct perplexity formula (different from raw loss)
    perplexity = 0
    count = 0
    #len(test_labels) // model.batch_size
    #len(test_inputs)//model.batch_size
    for i in range(len(test_inputs)//model.batch_size):
        X_batch = tf.convert_to_tensor(test_inputs[i * model.batch_size: (i+1) * model.batch_size])
        Y_batch = tf.convert_to_tensor(test_labels[i * model.batch_size: (i+1) * model.batch_size])
        logits, final = model.call(X_batch, None)
        average_loss = model.loss(logits, Y_batch)
        perplexity += average_loss
        count += 1
    return np.e **(perplexity / count)


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    # NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = tf.convert_to_tensor([[first_word_index]])
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0, 0, :])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n]) / np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n, p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = tf.convert_to_tensor([[out_index]])

    print(" ".join(text))


def main():
    # TODO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep,
    # so you want to remove the last element from train_x and test_x.
    # You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see impossibly small perplexities.
    
    # TODO: Separate your train and test data into inputs and labels
    window = 20
    train_data, test_data, word2id = get_data('data/train.txt', 'data/test.txt')
    # TO-DO:  Separate your train and test data into inputs and labels
    input = train_data.copy()
    input.pop(0)
    label = train_data.copy()
    label.pop(len(train_data) - 1)
    
    # TODO: initialize model and tensorflow variables
    model = Model(vocab_size = len(word2id))
    # TODO: Set-up the training step
    for i in range(model.epoch):
        train(model,input, label)
    # TODO: Set up the testing steps
    input = test_data.copy()
    input.pop(0)
    label = test_data.copy()
    label.pop(len(test_data) - 1)
    perplexity = test(model, input, label)
    # Print out perplexity
    print("perplexity is ", perplexity)
    # BONUS: Try printing out sentences with different starting words
    generate_sentence("john", 20, word2id, model)
    # BONUS: Try printing out various sentences with different start words and sample_n parameters
    pass


if __name__ == "__main__":
    main()
