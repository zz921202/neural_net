from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FALGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('keep_prob', 0.5, """keep probability for dropout""")


def weight_variable(shape):
    seed_val = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(seed_val)


def bias_variable(shape):
    seed_val = tf.constant(0.1, shape=shape)
    return tf.Variable(seed_val)


def conv2d(x, W, stride=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')


def max_pool(x, pool_name, filter_size=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.name_scope(pool_name):
        out = tf.nn.max_pool(x, filter_size, strides=strides, padding='SAME')
        variable_summaries(pool_name, out)
    return out


def variable_summaries(name, var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def simple_nn_layer(input_tensor, input_dim, out_dim,
                    layer_name, dropout=False, stride_size=[1, 1, 1, 1]):
    """ 
    input tensor of dimension : batch_size x height x width x channels
    input_dim : width x height of input filter
    out_dim : [input_channel, output_channel]
    layer_name: a string

    return : output tensor
    """

    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            w = weight_variable(input_dim + out_dim)
            variable_summaries(layer_name + '/weights', w)
        with tf.name_scope('biases'):
            outb = out_dim[1]
            b = bias_variable([outb])
            variable_summaries(layer_name + '/bias', b)
        with tf.name_scope('activation'):
            ac = conv2d(input_tensor, w, stride=stride_size) + b
            variable_summaries(layer_name + '/activation', ac)
        with tf.name_scope('relu'):
            out_tensor = tf.nn.relu(ac)
            variable_summaries(layer_name + '/relu', out_tensor)
        if dropout:
            return tf.nn.dropout(out_tensor, FALGS.keep_prob)
        return out_tensor


def simple_fully_connected_layer(input_tensor, feature_len, out_len, layer_name,
                                 relu=True, dropout=False):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            w = weight_variable([feature_len, out_len])
            variable_summaries(layer_name + '/weights', w)
        with tf.name_scope('biases'):
            b = bias_variable([out_len])
            variable_summaries(layer_name + '/bias', b)
        if relu:
            with tf.name_scope('relu_activation'):
                out_tensor = tf.nn.relu(tf.matmul(input_tensor, w) + b)
                variable_summaries(layer_name + '/relu_ac', out_tensor)
        else:
            with tf.name_scope('activation'):
                out_tensor = tf.matmul(input_tensor, w) + b
                variable_summaries(layer_name + '/relu_ac', out_tensor)
        if dropout:
            out_tensor = tf.nn.dropout(out_tensor, FALGS.keep_prob)
        return out_tensor

# def cross_entropy_layer()
# setup the graph
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


#  ARCHITECHUAL CNN

x_image = tf.reshape(x, [-1, 28, 28, 1])

# conv layer 1
conv1 = simple_nn_layer(x_image, [5, 5], [1, 32], 'conv1')
conv1_p = max_pool(conv1, 'conv1_p')
# conv layer 2
conv2 = simple_nn_layer(conv1_p, [5, 5], [32, 64], 'conv2')
conv2_p = max_pool(conv2, 'conv2_p')

# unroll vector
fc1_flat = tf.reshape(conv2_p, [-1, 7 * 7 * 64])

# 3 fully connected layer
fc1 = simple_fully_connected_layer(fc1_flat, 7 * 7 * 64, 1024,
                                   'fc1', dropout=True)
fc2 = simple_fully_connected_layer(fc1, 1024, 10, 'fc2',
                                   relu=False)

# evaluate the network and train
y_soft = tf.nn.softmax(fc2)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_soft))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_soft, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.initialize_all_variables())
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0], y_: batch[1]})
        print('step %d with accuracy %s' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


FALGS.keep_prob = 1
print('test accuracy %s' % accuracy.eval(
    feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
