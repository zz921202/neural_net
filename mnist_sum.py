from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from os import makedirs
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('keep_prob', 0.5, """keep probability for dropout""")

RESTORE = True
Restore_path = "./mnist/model100.ckpt"


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
        # tf.scalar_summary('max/' + name, tf.reduce_max(var))
        # tf.scalar_summary('min/' + name, tf.reduce_min(var))
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
            return tf.nn.dropout(out_tensor, FLAGS.keep_prob)
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
            out_tensor = tf.nn.dropout(out_tensor, FLAGS.keep_prob)
        return out_tensor


def soft_to_label_accruarcy(
        logits, target, layer_name="accuracy", indicator=False):
    """NOTE: target should be labels in the format of {0,1,2,3,4}
    ! not indicator"""
    if indicator:
        target = tf.argmax(target, 1)
    # loss_ave = tf.train.ExponentialMovingAverage(0.9, name='ave')

    # accuracy = tf.Variable(initial_value=tf.zeros([1]))
    # accuracy_update_op = loss_ave.apply(accuracy)
    # shadow_acc = loss_ave.average(accuracy)
    # with tf.name_scope(layer_name):
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int64),
                                  tf.cast(target, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)
    # tf.scalar_summary('accuracy_ave', shadow_acc)
    return accuracy


def cross_entropy_layer(logits, target, layer_name='cross_entropy', indicator=False):
    if indicator:
        cross_entropy = -tf.reduce_sum(target * tf.log(logits))
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, target, name='cross_entropy_per_example')

    with tf.name_scope(layer_name):
        tf.scalar_summary('cross_entropy', cross_entropy)
    return cross_entropy

def detect_dir_and_delete(dir_name):
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)



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
size = 7 * 7 * 64
fc1_flat = tf.reshape(conv2_p, [-1, size])

# 3 fully connected layer
fc1 = simple_fully_connected_layer(fc1_flat, size, 1024,
                                   'fc1', dropout=True)
fc2 = simple_fully_connected_layer(fc1, 1024, 10, 'fc2',
                                   relu=False)

# evaluate the network and train
y_soft = tf.nn.softmax(fc2)
cross_entropy = cross_entropy_layer(y_soft, y_, indicator=True)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

accuracy = soft_to_label_accruarcy(y_soft, y_, indicator=True)

# correct_prediction = tf.equal(tf.argmax(y_soft, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# summary ops
# makedirs('./mnist_log')
curdir = './mnist_log'
detect_dir_and_delete(curdir)


saver = tf.train.Saver()

summary_writer = tf.train.SummaryWriter(
    curdir, graph_def=sess.graph_def)
summary_op = tf.merge_all_summaries()

# initialize all variables
sess.run(tf.initialize_all_variables())

if RESTORE:
    saver.restore(sess, Restore_path)
    print("Model restored.")


for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        # sess.run(accuracy_update_op)
        sum_str = summary_op.eval(feed_dict={x: batch[0], y_: batch[1]})
        summary_writer.add_summary(sum_str, i)
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0], y_: batch[1]})
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0], y_: batch[1]})
        print('step %d with accuracy %s' % (i, train_accuracy))

    if i % 1000 == 0:
        save_path = saver.save(sess, "./mnist/model%s.ckpt" % i)
        print("Model saved in file: %s" % save_path)

    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


FLAGS.keep_prob = 1
print('test accuracy %s' % accuracy.eval(
    feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
