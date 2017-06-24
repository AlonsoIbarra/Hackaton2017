import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# Recognize handwritten digits from data set of mnist

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# Three hidden layers 500 neurons each
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10    # Only 10 digits exists 0..9
batch_size = 100  # Get batches to not to load all on memory

# 28x28 pixel images = 784 values
x = tf.placeholder('float', [None, 784])  # No hight x 784 width
y = tf.placeholder('float')


def neural_network_model(data):
    """
    :param data: The input data.
    """
    # Shape for weights 784 * nodes number in the hidden layer 1
    weights_tensor_l1 = tf.random_normal([784, n_nodes_hl1])

    # Shape for biases (the number of nodes in hidden layer 1)
    biases_tensor_l1 = tf.random_normal([n_nodes_hl1])

    # Other layers tensor shapes
    weights_tensor_l2 = tf.random_normal([n_nodes_hl1, n_nodes_hl2])
    biases_tensor_l2 = tf.random_normal([n_nodes_hl2])

    weights_tensor_l3 = tf.random_normal([n_nodes_hl2, n_nodes_hl3])
    biases_tensor_l3 = tf.random_normal([n_nodes_hl3])

    weights_tensor_l4 = tf.random_normal([n_nodes_hl3, n_classes])
    biases_tensor_l4 = tf.random_normal([n_classes])

    # Later refactor to use loops
    hidden_1_layer = {
        'weights': tf.Variable(weights_tensor_l1),
        'biases': tf.Variable(biases_tensor_l1)
    }

    hidden_2_layer = {
        'weights': tf.Variable(weights_tensor_l2),
        'biases': tf.Variable(biases_tensor_l2)
    }

    hidden_3_layer = {
        'weights': tf.Variable(weights_tensor_l3),
        'biases': tf.Variable(biases_tensor_l3)
    }

    output_layer = {
        'weights': tf.Variable(weights_tensor_l4),
        'biases': tf.Variable(biases_tensor_l4)
    }

    # (input_data * weights) + biases

    l1 = tf.add(
        tf.matmul(data, hidden_1_layer['weights']),
        hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)  # Activation function

    l2 = tf.add(
        tf.matmul(l1, hidden_2_layer['weights']),
        hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(
        tf.matmul(l2, hidden_3_layer['weights']),
        hidden_3_layer['biases'])

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    difference = tf.nn.softmax_cross_entropy_with_logits(prediction, y)
    cost = tf.reduce_mean(difference)

    # learning_rate = 0.001 by default
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10  # Do feed foward and backward propagation 10 times

    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            # Cycle "the number of batches" times
            for _ in range(int(mnist.train.num_examples / batch_size)):
                # Train a batch
                epx, epy = mnist.train.next_batch(batch_size)
                _, the_cost = sess.run(
                    [optimizer, cost],
                    feed_dict={x: epx, y: epy})

                epoch_loss += the_cost

            print('Epoch {} completed of out {} | Loss: {}'.format(
                epoch,
                hm_epochs,
                epoch_loss))

        # Assert prediction is equal to the training example
        correct = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc = accuracy.eval({
            x: mnist.test.images,
            y: mnist.test.labels})

        print('Accuracy: ', acc)


if __name__ == '__main__':
    train_neural_network(x)
