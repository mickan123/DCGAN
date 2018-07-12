import tensorflow as tf
from tensorflow.python.layers import base

#Coord conv from paper https://arxiv.org/pdf/1807.03247.pdf

class AddCoords(base.Layer):

	"""Add coords to a tensor"""
	def __init__(self, x_dim=96, y_dim=96, with_r=False):
		super(AddCoords, self).__init__()
		self.x_dim = x_dim
		self.y_dim = y_dim
		self.with_r = with_r
	
	def call(self, input_tensor):

		"""
		input_tensor: (batch, x_dim, y_dim, c)
		"""
		batch_size_tensor = tf.shape(input_tensor)[0]
		xx_ones = tf.ones([batch_size_tensor, self.x_dim], dtype=tf.int32)
		xx_ones = tf.expand_dims(xx_ones, -1)
		xx_range = tf.tile(tf.expand_dims(tf.range(self.x_dim), 0), [batch_size_tensor, 1])
		xx_range = tf.expand_dims(xx_range, 1)
		xx_channel = tf.matmul(xx_ones, xx_range)
		xx_channel = tf.expand_dims(xx_channel, -1)
		yy_ones = tf.ones([batch_size_tensor, self.y_dim], dtype=tf.int32)
		yy_ones = tf.expand_dims(yy_ones, 1)
		yy_range = tf.tile(tf.expand_dims(tf.range(self.y_dim), 0), [batch_size_tensor, 1])
		yy_range = tf.expand_dims(yy_range, -1)
		yy_channel = tf.matmul(yy_range, yy_ones)
		yy_channel = tf.expand_dims(yy_channel, -1)
		xx_channel = tf.cast(xx_channel, 'float32') / (self.x_dim - 1)
		yy_channel = tf.cast(yy_channel, 'float32') / (self.y_dim - 1)
		xx_channel = xx_channel*2 - 1
		yy_channel = yy_channel*2 - 1
		ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)
		return ret


def CoordConv(input_tensor, *args, **kwargs):
	addcoords = AddCoords()
	coords_added_tensor = addcoords(input_tensor)
	conv = tf.layers.conv2d(coords_added_tensor, *args, **kwargs)
	return conv