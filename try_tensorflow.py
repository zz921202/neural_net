import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
myinput = - np.array(range(1, 101)).reshape(1,10,10,1)

print(myinput.__str__())
x = tf.placeholder(tf.float32)
max_pool = tf.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

sess.run(tf.initialize_all_variables())

print(max_pool.eval(feed_dict={x: myinput}))
