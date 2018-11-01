import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys
import DataProcess as DP


# Load and transform data first
print("Loading Data")
dataI, dataQ, labels = DP.loadData()

scale = float(sys.argv[1])

print("Converting")
trainData, trainLabel, testData, testLabel = DP.convertData(
    dataI, dataQ, labels, scale)


# Training Parameters
learning_rate = 0.001
training_steps = int(sys.argv[2]) + 1  # input
batch_size = 32
display_step = 100

# Network Parameters
num_input = 240
timesteps = 50  # timesteps
num_lstmcell = 50
num_classes = testLabel.shape[1]
num_samples = trainData.shape[0]

print("Finished loading data!")


def RNN(x, weights, biases):

    # hidden layer
    # x = tf.reshape(x, [-1, num_input])
    # x = tf.matmul(x, weights['w1']) + biases['w1']
    # x = tf.nn.relu(x)

    # x = tf.reshape(x, [-1, timesteps, num_hidden])

    x_in = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
        num_lstmcell, forget_bias=1.0, state_is_tuple=True)

    # static
    outputs, states = tf.nn.static_rnn(lstm_cell, x_in, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


index = 0


def next_batch(batch_size):
    global index
    global trainData
    global trainLabel
    start = index
    index += batch_size

    if index > num_samples:
        # Finished epoch
        perm = np.arange(num_samples)
        np.random.shuffle(perm)
        trainData = trainData[perm]
        trainLabel = trainLabel[perm]

        # Next epoch
        start = 0
        index = batch_size
        assert batch_size < num_samples

    end = index
    return trainData[start:end], trainLabel[start:end]


# tf Graph input
X = tf.placeholder("float32", [None, timesteps, num_input])
Y = tf.placeholder("float32", [None, num_classes])

# Initial state of the LSTM memory.
# init_state = tf.placeholder("float32", [None, 2 * num_lstmcell])

# Define weights
weights = {
    #'w1': tf.Variable(tf.random_normal([num_input, num_hidden])),
    'out': tf.Variable(tf.random_normal([num_lstmcell, num_classes]))
}
biases = {
    #'w1': tf.Variable(tf.random_normal([num_hidden])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Softmax loss
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# Training
with tf.Session() as sess:

    # Train
    print("Training")
    # Run the initializer
    sess.run(init)

    # Default
    for step in range(1, training_steps):
        batch_x, batch_y = next_batch(batch_size)
        # Reshape data to 50x200
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            _loss, acc = sess.run(
                [loss, accuracy], feed_dict={X: batch_x,
                                             Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(
                _loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

        if step % (display_step * 5) == 0 or step == 1:
            test_len = 4000
            test_data = testData[:test_len].reshape((-1, timesteps, num_input))
            test_label = testLabel[:test_len]

            cp, accu = sess.run(
                [correct_pred, accuracy],
                feed_dict={X: test_data,
                           Y: test_label})
            print("Testing Accuracy:", accu)

            with open("Trainsteps-NoiseCurve.txt", 'a') as f:
                f.write(str(sys.argv[1]))
                f.write(" ")
                f.write(str(step))
                f.write(" ")
                f.write(str(accu))
                f.write("\n")
                f.close()

    print("Optimization Finished!")
