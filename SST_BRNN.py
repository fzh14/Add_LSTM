import tensorflow as tf
import gensim
import numpy as np
import random

##### prepare data
path = 'stanfordSentimentTreebank/output_dataset.txt'
model_path = 'stanfordSentimentTreebank/output'
model = gensim.models.Word2Vec.load(model_path)

sentence_max = 56

class Data(object):
    def __init__(self):
        self.data_in = []
        self.data_label = []
        self.batch_id = 0
        fp = open(path, 'r')
        for l in fp.readlines():
            line = l.strip('\n').split('|')
            word_list = line[0].split(' ')
            s = []
            for item in word_list:
                s.append(model.wv[item].tolist())
            if len(word_list)<sentence_max:
                for i in range(sentence_max-len(word_list)):
                    s.append([0. for k in range(100)])
            l = [0. for k in range(5)]
            value = float(line[1])
            label_index = int(value / 0.2)
            if label_index >= 5:
                l[4] = 1.0
            else:
                l[label_index] = 1.0
            if len(s[random.randint(0,sentence_max-1)]) != 100:
                print 'error'
            self.data_in.append(s)
            self.data_label.append(l)

    def next(self, batch_size):
        if self.batch_id + batch_size >= len(self.data_in):
            batch_data_in = self.data_in[self.batch_id : len(self.data_in)]
            batch_data_label = self.data_label[self.batch_id : len(self.data_in)]
            self.batch_id = self.batch_id + batch_size - len(self.data_in)
            batch_data_in += self.data_in[0:self.batch_id]
            batch_data_label += self.data_label[0:self.batch_id]
        else:
            batch_data_in = self.data_in[self.batch_id : self.batch_id+batch_size]
            batch_data_label = self.data_label[self.batch_id : self.batch_id+batch_size]
            self.batch_id = self.batch_id + batch_size
        return batch_data_in, batch_data_label

trainset = Data()

# ==============
#     MODEL
# ==============

learning_rate = 0.001
training_iters = 500000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 100 # data input (shape: 100*56)
n_steps = 56 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 5 # total classes

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    # (2*128, 5)
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes], ))
}

def BiRNN(x, weights, biases):

    X_in = tf.unstack(x, n_steps, 1)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, X_in,
                                                 dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = BiRNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 1
    while step * batch_size <= training_iters:
        batch_x, batch_y = trainset.next(batch_size)
        sess.run(optimizer, feed_dict={
            x:batch_x,
            y:batch_y
        })
        if step%display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    print 'Optimizer Complete'