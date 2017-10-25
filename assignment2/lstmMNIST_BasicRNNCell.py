# Most code inspired from
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
import tensorflow as tf
import os
#from tensorflow.python.ops import rnn, rnn_cell, rnn_cell_impl
from tensorflow.contrib import rnn  # copied from Damien Aymeric
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

result_dir='results/LSTM/RNN/2/'
# call mnist function
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learningRate = 1e-4
trainingIters = 50000
batchSize = 100
displayStep = 20

nInput = 28  # we want the input to take the 28 pixels
nSteps = 28  # every 28
nHidden = 128  # number of neurons for the RNN
nClasses = 10  # this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases):
    # current code from Damien Aymeric
    x = tf.unstack(x, nSteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(nHidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# optimization
# create the cost, optimization, evaluation, and accuracy
# for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# Add a scalar summary for the snapshot loss.
tf.summary.scalar(cost.op.name, cost)
tf.summary.scalar('accuracy', accuracy)  # to plot train accuracy

# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.summary.merge_all()

# Add the variable initializer Op.
init = tf.global_variables_initializer()  # sess.run()?

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()




with tf.Session() as sess:
    sess.run(init)
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
    step = 1

    while (step * batchSize) < trainingIters:
        # print('hi, this is iteration ', step, 'of the while loop')
        batchX, batchY = mnist.train.next_batch(batchSize)  # mnist has a way to get the next batch
        batchX = batchX.reshape((batchSize, nSteps, nInput))

        sess.run(optimizer, feed_dict={x: batchX, y: batchY})

        if step % displayStep == 0:
            acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
            loss = sess.run(cost, feed_dict={x: batchX, y: batchY})
            print("Iter " + str(step * batchSize) +
                  ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
            # Update the events file which is used to monitor the training (in this case,
            # only the training loss is monitored)
            summary_str = sess.run(summary_op, feed_dict={x: batchX, y: batchY})
            summary_writer.add_summary(summary_str, step*batchSize)
            summary_writer.flush()
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=step*batchSize)
        step += 1
    print('Optimization finished')

    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: testData, y: testLabel}))