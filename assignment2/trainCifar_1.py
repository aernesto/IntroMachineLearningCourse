from scipy import misc
import os
import time
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp

# --------------------------------------------------
# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(initial, name="W")

    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial, name="b")
    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')
    return h_max


ntrain = 1000  # per class
ntest = 100  # per class
nclass = 10  # number of classes
imsize = 28
nchannels = 1
batchsize = 100

# Load training and test images
Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = './CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        im = misc.imread(path)  # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain, :, :, 0] = im
        LTrain[itrain, iclass] = 1  # 1-hot label
    for isample in range(0, ntest):
        path = './CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
        im = misc.imread(path)  # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest, :, :, 0] = im
        LTest[itest, iclass] = 1  # 1-hot label

sess = tf.InteractiveSession()
start_time = time.time()  # start timing
#tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels], name="x")
#tf_data = tf.reshape(x, [-1, 28, 28, 1])
#tf variable for labels
tf_labels = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

result_dir = './results/4/'  # directory where the results from the training are saved
max_step = 10000  # the maximum iterations. After max_step iterations, the training will stop no matter what

# --------------------------------------------------
# model
#create your model

# first convolutional layer
with tf.name_scope("conv1"):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(tf_data, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    tf.summary.histogram("weights", W_conv1)
    tf.summary.histogram("biases", b_conv1)
    tf.summary.histogram("ReLu_activations", h_conv1)
    tf.summary.histogram("max-pool_activations", h_pool1)

# second convolutional layer
with tf.name_scope("conv2"):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    tf.summary.histogram("weights", W_conv2)
    tf.summary.histogram("biases", b_conv2)
    tf.summary.histogram("ReLu_activations", h_conv2)
    tf.summary.histogram("max-pool_activations", h_pool2)


# densely connected layer
with tf.name_scope("fc1"):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    tf.summary.histogram("weights", W_fc1)
    tf.summary.histogram("biases", b_fc1)
    tf.summary.histogram("ReLu_activations", h_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second fully connected
with tf.name_scope("fc2"):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    tf.summary.histogram("weights", W_fc2)
    tf.summary.histogram("biases", b_fc2)
# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy
# setup training
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=y_conv))
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add a scalar summary for the snapshot loss.
tf.summary.scalar(cross_entropy.op.name, cross_entropy)
tf.summary.scalar('accuracy', accuracy)  # to plot train accuracy

# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.summary.merge_all()

# Add the variable initializer Op.
init = tf.global_variables_initializer()  # sess.run()?

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Instantiate a SummaryWriter to output summaries and the Graph.
summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

# --------------------------------------------------
# optimization

sess.run(init)
print("test accuracy before training: %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))
#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_xs = np.zeros([batchsize, imsize, imsize, nchannels])
#setup as [batchsize, the number of many classes]
batch_ys = np.zeros([batchsize, nclass])
nsamples = Train.shape[0]
niter = max_step  # try a small iteration size once it works then continue
for i in range(niter):
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j, :, :, :] = Train[perm[j], :, :, :]
        batch_ys[j, :] = LTrain[perm[j], :]
    if i % 100 == 0:
        #calculate train accuracy and print it
        # run the training
        # output the training accuracy every 10 iterations
        train_accuracy = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        if i == 0:
            print("step %d, training accuracy before training: %g" % (i, train_accuracy))
        elif i == niter - 1:
            print("step %d, training accuracy at end of training: %g" % (i, train_accuracy))

        # Update the events file which is used to monitor the training (in this case,
        # only the training loss is monitored)
        summary_str = sess.run(summary_op, feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

    # save the checkpoints every 1100 iterations
    if i % 1100 == 0 or i == niter - 1:
        checkpoint_file = os.path.join(result_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=i)
        # print test error
        #print("test accuracy %g" % accuracy.eval(feed_dict={
        #    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    train_step.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})  # run one train_step

stop_time = time.time()
print('The training takes %f second to finish' % (stop_time - start_time))


    #optimizer.run(feed_dict={}) # dropout only during training

# --------------------------------------------------
# test

print("test accuracy after training: %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))

sess.close()
