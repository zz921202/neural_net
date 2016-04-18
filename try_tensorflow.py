from __future__ import print_function
import tensorflow as tf
import numpy as np

var0 = tf.Variable(initial_value=tf.zeros([1]))
var1 = tf.Variable(initial_value=tf.zeros([1]))
sess = tf.InteractiveSession()
# myinput = - np.array(range(1, 101)).reshape(1, 10, 10, 1)
# print(myinput.__str__())
# x = tf.placeholder(tf.float32)

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


loss_ave = tf.train.ExponentialMovingAverage(0.9, name='avg')
loss_averages_op = loss_ave.apply([var0])
shadow_var = loss_ave.average(var0)
# max_pool = tf.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1],
#                           padding='SAME')
y = tf.constant(6, dtype=tf.float32)
add_op = tf.add(var0, y)
update = tf.assign(var0, add_op)

variable_summaries('hi', var0)
summary_writer = tf.train.SummaryWriter(
    './mnist_log', graph_def=sess.graph_def)
summary_op = tf.merge_all_summaries()

sess.run(tf.initialize_all_variables())
# print(max_pool.eval(feed_dict={x  : myinput}))
for i in range(10):

    print(sess.run(update))
    print(sess.run(shadow_var))
    sumstr = sess.run(summary_op)
    summary_writer.add_summary(sumstr, i)

