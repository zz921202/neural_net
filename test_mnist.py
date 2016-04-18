from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# NAIVE softmax implementation

def weight_variable(shape):
    seed_val = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(seed_val)


def bias_variable(shape):
    seed_val = tf.constant(0.1, shape=shape)
    return tf.Variable(seed_val)



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


def simple_fully_connected_layer(input_tensor, feature_len, out_len, layer_name,
                                 relu=False, dropout=False):
    with tf.name_scope(layer_name):
        w = weight_variable([feature_len, out_len])
        b = bias_variable([out_len])
        # with tf.name_scope('weights'):
        #     w = weight_variable([feature_len, out_len])
        #     variable_summaries(layer_name + '/weights', w)
        # with tf.name_scope('biases'):
        #     b = bias_variable([out_len])
        #     variable_summaries(layer_name + '/bias', b)
        if relu:
            with tf.name_scope('relu_activation'):
                out_tensor = tf.nn.relu(tf.matmul(input_tensor, w) + b)
                variable_summaries(layer_name + '/relu_ac', out_tensor)
        else:
            with tf.name_scope('activation'):
                out_tensor = tf.matmul(input_tensor, w) + b
                variable_summaries(layer_name + '/relu_ac', out_tensor)
        # if dropout:
        #     out_tensor = tf.nn.dropout(out_tensor, FLAGS.keep_prob)
        return out_tensor


w = tf.Variable(initial_value=tf.zeros([784, 10]))
bias = tf.Variable(initial_value=tf.zeros([10]))

x_image = tf.reshape(x, [-1, 28 * 28 * 1])

out = simple_fully_connected_layer(x_image, 28*28, 10, 'hello')


y = tf.nn.softmax(out)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


summary_writer = tf.train.SummaryWriter(
    './mnist_log', graph_def=sess.graph_def)
summary_op = tf.merge_all_summaries()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 1 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0], y_: batch[1]})
        sum_str = summary_op.eval(
            feed_dict={x: batch[0], y_: batch[1]})
        summary_writer.add_summary(sum_str, i)
        print('step %d with accuracy %s' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


print('test accuracy %s' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


