import tensorflow as tf
import gensim
import string
import numpy as np
import random

##### prepare data
path = 'stanfordSentimentTreebank/output_50d.txt'
# model_path = 'stanfordSentimentTreebank/output'
# model = gensim.models.Word2Vec.load(model_path)
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/ivanfzh/Downloads/glove.6B/glove.6B.50d.txt',
                                                        binary=False)

sentence_max = 56


class Data(object):
    def __init__(self):
        self.data_in = []
        self.data_label = []
        self.batch_id = 0
        self.data_length = []

        fp = open(path, 'r')
        for l in fp.readlines():
            line = l.strip('\n').split('|')
            word_list = line[0].split(' ')
            s = []
            for item in word_list:
                item = string.lower(item)
                s.append(model[item].tolist())
            if len(word_list) < sentence_max:
                for i in range(sentence_max - len(word_list)):
                    s.append([0. for k in range(50)])
            self.data_length.append(len(word_list))
            l = [0. for k in range(5)]
            value = float(line[1])
            label_index = int(value / 0.2)
            if label_index >= 5:
                l[4] = 1.0
            else:
                l[label_index] = 1.0
            self.data_in.append(s)
            self.data_label.append(l)

    def next(self, batch_size):
        if self.batch_id + batch_size >= len(self.data_in):
            batch_data_in = self.data_in[self.batch_id: len(self.data_in)]
            batch_data_label = self.data_label[self.batch_id: len(self.data_in)]
            batch_data_length = self.data_length[self.batch_id: len(self.data_in)]
            self.batch_id = self.batch_id + batch_size - len(self.data_in)
            batch_data_in += self.data_in[0:self.batch_id]
            batch_data_label += self.data_label[0:self.batch_id]
            batch_data_length += self.data_length[0:self.batch_id]
        else:
            batch_data_in = self.data_in[self.batch_id: self.batch_id + batch_size]
            batch_data_label = self.data_label[self.batch_id: self.batch_id + batch_size]
            batch_data_length = self.data_length[self.batch_id: self.batch_id + batch_size]
            self.batch_id = self.batch_id + batch_size
        return batch_data_in, batch_data_label, batch_data_length


trainset = Data()

print len(trainset.data_in)

# ==============
#     MODEL
# ==============

learning_rate = 0.001
training_iters = 500000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 50  # data input (shape: 50*56)
n_steps = 56  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 5  # total classes

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
z = tf.placeholder(tf.int32, [batch_size])

weights = {
    # (50, 128)
    #   'in': tf.Variable(tf.random_normal([n_input, n_hidden])),
    # Hidden layer weights
    # (128, 5)
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    #  'in': tf.Variable(tf.constant(0.1, shape=[n_hidden, ])),
    'out': tf.Variable(tf.random_normal([n_classes, ]))
}


def dynamicRNN(x, seqlen, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, sentence_max, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * sentence_max + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']


pred = dynamicRNN(x, z, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

t_acc = 0

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 1
    while step * batch_size <= training_iters:
        batch_x, batch_y, batch_length = trainset.next(batch_size)
        sess.run(optimizer, feed_dict={
            x: batch_x,
            y: batch_y,
            z: batch_length
        })
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, z: batch_length})
        t_acc = (acc + t_acc * (step - 1)) / (float(step))
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, z: batch_length})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, z: batch_length})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(t_acc) + ",batch Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    print 'Optimizer Complete'
