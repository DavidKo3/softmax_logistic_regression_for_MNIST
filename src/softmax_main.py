import tensorflow as tf
import input_data

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)

# tensorflow graph input
X = tf.placeholder('float', [None, 784]) # mnist data image of shape 28 * 28 = 784
Y = tf.placeholder('float', [None, 10]) # 0-9 digits recognition = > 10 classes

# set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Our hypothesis
activation = tf.add(tf.matmul(X, W),b)  # Softmax

# Cost function: cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activation, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Gradient Descent

# Before starting, initialize the variables. We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)







