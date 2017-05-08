import tensorflow as tf
import numpy as np
dim = 512
n = 10
centers = tf.get_variable('centers', [n, dim], dtype = tf.float32, initializer = tf.constant_initializer(0), trainable = False)
label = np.array([0,1,2,3,4,4,3,2,1,0,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1])
label  = tf.reshape(label, [-1])
centers_batch = tf.gather(centers, label)
print(centers_batch)
with tf.Session(''):
    print(label.eval().shape)
    print(centers_batch.eval())

