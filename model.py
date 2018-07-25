import tensorflow as tf
from ops import lrelu
from CoordConv import CoordConv

#Repeated block to be used in the discriminator
def discriminator_block(inputs, out, kernel_size, stride):
	res = inputs
	net = tf.layers.conv2d(inputs, out, kernel_size, stride, padding = 'same', activation = None)
	net = lrelu(net)
	net = tf.layers.conv2d(net, out, kernel_size, stride, padding = 'same', activation = None)
	net = net + inputs
	net = lrelu(net)
	return net

#Creates the discriminator network
def discriminator(x, image_dim, mult, isTrain = False, reuse = False, coord_conv = False):

	x_image = tf.reshape(x, [-1, image_dim, image_dim, 3])

	with tf.variable_scope('Discriminator', reuse = reuse):

		if (coord_conv):
			net = CoordConv(x_image, int(32 * mult), 3, 1, padding = 'same')
		else:
			net = tf.layers.conv2d(x_image, int(32 * mult), 3, 1, padding = 'same')

		net = lrelu(net)

		net = discriminator_block(net, int(32 * mult), 3, 1)

		net = tf.layers.conv2d(net, int(64 * mult), 4, 2, padding = 'same')
		net = lrelu(net)

		for _ in range(4):
			net = discriminator_block(net, int(64 * mult), 3, 1)

		net = tf.layers.conv2d(net, int(128 * mult), 4, 2, padding = 'same')
		net = lrelu(net)

		for _ in range(4):
			net = discriminator_block(net, int(128 * mult), 3, 1)

		net = tf.layers.conv2d(net, int(256 * mult), 3, 2, padding = 'same')
		net = lrelu(net)

		for _ in range(4):
			net = discriminator_block(net, int(256 * mult), 3, 1)

		net = tf.layers.conv2d(net, int(512 * mult), 3, 2, padding = 'same')
		net = lrelu(net)

		for _ in range(4):
			net = discriminator_block(net, int(512 * mult), 3, 1)

		net = tf.layers.conv2d(net, int(512 * mult), 3, 2, padding = 'same')
		net = lrelu(net)

		net = tf.layers.Flatten()(net)
		logits = tf.layers.dense(net, 1)
		tags = tf.layers.dense(net, 23)

	return logits, tags

#Repeated block to be used in the generator
def generator_block(inputs, out, kernel_size, stride, isTrain = True):

	net = tf.layers.conv2d_transpose(inputs, out, kernel_size, stride, padding = 'same')
	net = tf.layers.batch_normalization(net, training = isTrain)
	net = tf.nn.relu(net)
	net = tf.layers.conv2d_transpose(inputs, out, kernel_size, stride, padding = 'same')
	net = tf.layers.batch_normalization(net, training = isTrain)
	net = net + inputs

	return net

#Creates the generator network
def generator(z, mult, tags = None, isTrain = True, reuse = False):

	with tf.variable_scope('Generator', reuse = reuse):
		if tags != None:
			z = tf.concat([z, tags], axis = 1)

		net = tf.layers.dense(z, int(64*24*24 * mult))
		net = tf.layers.batch_normalization(net, training = isTrain)
		net = tf.reshape(net, [-1, 24, 24, int(64 * mult)])
		net = tf.nn.relu(net)

		inputs = net

		for i in range(16):
			net = generator_block(net, int(64 * mult), 3, 1, isTrain)

		net = tf.layers.batch_normalization(net, training = isTrain)
		net = tf.nn.relu(net)
		net = inputs + net

		net = tf.layers.conv2d_transpose(net, int(256 * mult), 3, 2, padding = 'same')
		net = tf.layers.batch_normalization(net, training = isTrain)
		net = tf.nn.relu(net)

		net = tf.layers.conv2d_transpose(net, int(256 * mult), 3, 2, padding = 'same')
		net = tf.layers.batch_normalization(net, training = isTrain)
		net = tf.nn.relu(net)

		net = tf.layers.conv2d_transpose(net, 3, 3, 1, padding = 'same')

		output = tf.nn.tanh(net)

	return output	