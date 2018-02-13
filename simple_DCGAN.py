def discriminator(x, isTrain = False, reuse = False):
	
	with tf.variable_scope('Discriminator', reuse = tf.AUTO_REUSE):
		conv1 = tf.layers.conv2d(x, 32, [5, 5], strides = (1,1), padding = 'same')
		lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training = isTrain))
		pool1 = tf.nn.max_pool(lrelu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		
		conv2 = tf.layers.conv2d(pool1, 64, [5, 5], strides = (1,1), padding = 'same')
		lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training = isTrain))
		pool2 = tf.nn.max_pool(lrelu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		pool2_flat = tf.reshape(pool2, [-1, 24*24*64])

		fc1 = tf.layers.dense(pool2_flat, 1)
		D_prob = tf.nn.sigmoid(fc1)

	return D_prob, fc1


def generator(z, isTrain = True, reuse = False):
	with tf.variable_scope('Generator', reuse = reuse):

		fc1 = tf.layers.dense(z, units = 24 * 24 * 64)
		lrelu1 = tf.nn.relu(fc1)
		relu_reshaped = tf.reshape(lrelu1, shape=[-1, 24, 24, 64])

		conv1 = tf.layers.conv2d_transpose(relu_reshaped, 128, [5, 5], strides = (2,2), padding = 'same')
		lrelu2 = lrelu(tf.layers.batch_normalization(conv1, training = isTrain))

		conv2 = tf.layers.conv2d_transpose(lrelu2, 3, [5, 5], strides = (2,2), padding = 'same')

		output = tf.nn.tanh(conv2)

	return output

