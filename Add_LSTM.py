import tensorflow as tf
import random

class AddDataSet(object):
    def __init__(self, n_samples=10000):
        self.data_in = []
        self.data_re = []
        for n in range(n_samples):
            a = random.randint(0,10000)
            b = random.randint(0,10000)
            while a+b >= 10000:
                a = random.randint(0, 10000)
                b = random.randint(0, 10000)
            result = a + b
            s = [[0. for k in range(10)] for m in range(8)]
            r = [0. for k in range(10)]
            for i in range(4):
                num_a = (a / (10**i)) % 10
                num_b = (b / (10**i)) % 10
                s[i][num_a] = 1.
                s[i+4][num_b] = 1.
            num_r = (result / (10**0)) % 10
            r[num_r] = 1.
            self.data_in.append(s)
            self.data_re.append(r)
        self.batch_id = 0

    def next(self,batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id + batch_size >= len(self.data_in):
            batch_data_in = self.data_in[self.batch_id : len(self.data_in)]
            batch_data_re = self.data_re[self.batch_id : len(self.data_in)]
            self.batch_id = self.batch_id + batch_size - len(self.data_in)
            batch_data_in += self.data_in[0:self.batch_id]
            batch_data_re += self.data_re[0:self.batch_id]
        else:
            batch_data_in = self.data_in[self.batch_id : self.batch_id+batch_size]
            batch_data_re = self.data_re[self.batch_id : self.batch_id+batch_size]
            self.batch_id = self.batch_id + batch_size
        return batch_data_in, batch_data_re


# ==========
#   MODEL
# ==========
lr = 0.001
training_iters = 1000000
batch_size = 128

n_inputs = 10   #data input (img shape: 10*8)
n_steps = 8    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      #classes (0-9 digits)

trainset = AddDataSet(n_samples=100000)
testset = AddDataSet(n_samples=500)

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    # (10, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # cell
    ##########################################
    # basic LSTM Cell.
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    ##outputs, states = tf.contrib.rnn.static_rnn(cell, X_in, dtype=tf.float32,
    ##                                            sequence_length=n_steps)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # shape [128, 10]
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_x, batch_y = trainset.next(batch_size)
        sess.run(train_op, feed_dict={
            x:batch_x,
            y:batch_y
        })
        if step % 500 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
