import tensorflow as tf
import random

class AddDataSet(object):
    def __init__(self, n_samples=10000):
        self.data_in = []
        self.data_re = []
        self.labels = []
        for n in range(n_samples):
            a = random.randint(0,10000)
            b = random.randint(0,10000)
            while a+b >= 10000:
                a = random.randint(0, 10000)
                b = random.randint(0, 10000)
            result = a + b
            s = [[0. for k in range(10)] for m in range(8)]
            r = [[0. for k in range(10)] for m in range(4)]
            l = []
            for i in range(4):
                num_a = (a / (10**i)) % 10
                num_b = (b / (10**i)) % 10
                s[i][num_a] = 1.
                s[i+4][num_b] = 1.
                num_r = (result / (10**0)) % 10
                r[i][num_r] = 1.
                l.append(num_r)
            self.data_in.append(s)
            self.data_re.append(r)
            self.labels.append(l)
        self.batch_id = 0

    def next(self,batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id + batch_size >= len(self.data_in):
            batch_data_in = self.data_in[self.batch_id : len(self.data_in)]
            batch_data_re = self.data_re[self.batch_id : len(self.data_in)]
            batch_labels = self.labels[self.batch_id : len(self.data_in)]
            self.batch_id = self.batch_id + batch_size - len(self.data_in)
            batch_data_in += self.data_in[0:self.batch_id]
            batch_data_re += self.data_re[0:self.batch_id]
            batch_labels += self.labels[0:self.batch_id]
        else:
            batch_data_in = self.data_in[self.batch_id : self.batch_id+batch_size]
            batch_data_re = self.data_re[self.batch_id : self.batch_id+batch_size]
            batch_labels = self.labels[self.batch_id : self.batch_id+batch_size]
            self.batch_id = self.batch_id + batch_size
        return batch_data_in, batch_data_re, batch_labels


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
#testset = AddDataSet(n_samples=500)

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, 4, n_classes])
y_label = tf.placeholder(tf.int64, [None, 4])

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

W = tf.Variable(tf.constant(1., shape=[batch_size, 4]))

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
    # [n_step, batch_size, n_hidden_units]
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # ==> [batch_size, 4, n_hidden_units]
    X_out = tf.transpose(outputs[-4:], [1,0,2])
    X_out = tf.reshape(X_out, [-1, n_hidden_units])
    # shape [128*4, 10]
    results = tf.matmul(X_out, weights['out']) + biases['out']
    # shape [128, 4, 10]
    results = tf.reshape(results, [-1, 4, n_classes])
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=pred,targets=y_label,weights=W))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,2), y_label)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_y_label = trainset.next(batch_size)
        sess.run(train_op, feed_dict={
            x:batch_x,
            y:batch_y,
            y_label:batch_y_label
        })
        if step % 200 == 0:
            acc = sess.run(accuracy, feed_dict={
                x:batch_x,
                y:batch_y,
                y_label:batch_y_label
            })
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={
                x: batch_x,
                y: batch_y,
                y_label:batch_y_label
            })
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
