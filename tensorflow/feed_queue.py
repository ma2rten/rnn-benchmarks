#!/usr/bin/env python
import time
import optparse
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn

def get_feed_dict(x_data, y_data=None):
    feed_dict = {}

    if not y_data is None:
        feed_dict[y] = y_data

    for i in xrange(X_data.shape[1]):
        feed_dict[x[i]] = x_data[:, i, :]

    return feed_dict

# Parameters
optparser = optparse.OptionParser()
optparser.add_option("-n", "--network_type", default='rnn', help="Network type (rnn, lstm, basic_lstm)")
optparser.add_option("-i", "--input_size", default=100, type='int', help="Input layer size")
optparser.add_option("-l", "--hidden_size", default=100, type='int', help="Hidden layer size")
optparser.add_option("-s", "--seq_length", default=30, type='int', help="Sequence length")
optparser.add_option("-b", "--batch_size", default=20, type='int', help="Batch size")
opts = optparser.parse_args()[0]

network_type = opts.network_type
input_size = opts.input_size
hidden_size = opts.hidden_size
seq_length = opts.seq_length
batch_size = opts.batch_size

n_samples = 100000

assert n_samples % batch_size == 0

# Data
X_data = np.random.rand(n_samples, seq_length, input_size).astype(np.float32)
Y_data = np.random.rand(n_samples, hidden_size).astype(np.float32)

targets_queue = tf.FIFOQueue(n_samples, "float", [seq_length, input_size])
input_queue = tf.FIFOQueue(n_samples, "float", [hidden_size])
batch = input_queue.dequeue_many(batch_size)

x = [batch[:, i, :] for i in xrange(seq_length)]
y = targets_queue.dequeue_many(batch_size)

if network_type == 'rnn':
    cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
elif network_type == 'lstm':
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, input_size)
elif network_type == 'basic_lstm':
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
else:
    raise Exception('Unknown network! '+network_type)

output, _cell_state = rnn.rnn(cell, x, dtype=tf.float32)
cost = tf.reduce_mean((output[-1] - y) ** 2)

optim = tf.train.GradientDescentOptimizer(0.01)
train_op = optim.minimize(cost)

session = tf.Session()
session.run(tf.initialize_all_variables())

session.run(inputs_queue.enqueue_many(X_data))

start = time.time()
for i in xrange(n_samples//batch_size):
    session.run(output[-1]) 

print "Forward:"
print "--- %i samples in %s seconds (%f samples/s) ---" % (n_samples, time.time() - start, n_samples / (time.time() - start))

session.run(inputs_queue.enqueue_many(X_data))
session.run(targets_queue.enqueue_many(X_data))

start = time.time()
for i in xrange(n_samples//batch_size):
    session.run(train_op, feed_dict={y: Y_data[i:i+batch_size]}) 

print "Forward + Backward:"
print "--- %i samples in %s seconds (%f samples/s) ---" % (n_samples, time.time() - start, n_samples / (time.time() - start))

session.close()
