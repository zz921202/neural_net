from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# train a simple softmax model
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

# define place holder to feed batched data into model

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# NAIVE softmax implementation

# w = tf.Variable(initial_value=tf.zeros([784, 10]))
# bias = tf.Variable(initial_value=tf.zeros([10]))

# sess.run(tf.initialize_all_variables())
# y = tf.nn.softmax(tf.matmul(x, w) + bias)
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# now build a CCN model for superior performance
# helper functions for initializing weights


def weight_variable(shape):
    seed_val = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(seed_val)


def bias_variable(shape):
    seed_val = tf.constant(0.1, shape=shape)
    return tf.Variable(seed_val)

# end of helper functions for weight initialization

# helper function for layers


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

# end of conv layer help functions

#  ARCHITECHUAL CNN

x_image = tf.reshape(x, [-1, 28, 28, 1])

# conv layer 1
w_con1 = weight_variable([5, 5, 1, 32])
b1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, w_con1))
h_pool1 = max_pool_2x2(h_conv1)

# conv layer 2
w_conv2 = weight_variable([5, 5, 32, 64])
b2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2))
h_pool1 = max_pool_2x2(h_conv2)

# 3 fully connected layer

w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1_flat = tf.reshape(h_pool1, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_fc1_flat, w_fc1) + b_fc1)

# drop out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# read out layer

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# END OF ARCHITECHTURE

# evaluation layer

cross_entropy = -tf.reduce_sum(y_conv * tf.log(y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print 'step %d with accuracy %s' % (i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


print 'test accuracy %s' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
