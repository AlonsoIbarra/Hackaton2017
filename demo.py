import tensorflow as tf

one = tf.constant(5)
two = tf.constant(6)

result = tf.mul(one, two)

print('Result is: ', result)

with tf.Session() as sess:
    output = sess.run(result)
    print(output)

print('Output is: ', output)
