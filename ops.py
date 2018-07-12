import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#Leaky relu function
def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

#Implements minibatch discrimination https://arxiv.org/abs/1606.03498
def minibatch_discrimination(input_layer, num_kernels, dim_per_kernel = 3):

	num_features = input_layer.shape[1]

	W = tf.get_variable('W', shape=[num_features, num_kernels*dim_per_kernel], initializer = tf.contrib.layers.xavier_initializer())
	b = tf.Variable(tf.zeros([num_kernels]))
	
	activation = tf.matmul(input_layer, W)
	activation = tf.reshape(activation, [-1, num_kernels, dim_per_kernel])

	diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, perm=[1,2,0]), 0)
	abs_diff = tf.reduce_sum(tf.abs(diffs), axis = 2)

	minibatch_features = tf.reduce_sum(tf.exp(-abs_diff), axis = 2)
	minibatch_features = minibatch_features + b

	return tf.concat([input_layer, minibatch_features], 1)


#Loads images dataset normalizing and resizing them
def load_images(size, path):
	image_list = []
	for i, image_path in enumerate(glob.glob(path + "/*")):
		image = misc.imread(image_path)
		image = misc.imresize(image, [size, size])
		image = image.flatten()/127.5 - 1
		image_list.append(image)
	
	return image_list

#Returns 4x4 plot of image samples
def plot(samples, dim):
	fig = plt.figure(figsize=(4, 4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in enumerate(samples):
		sample = (sample + 1) / 2
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(dim, dim, 3), cmap='Greys_r')

	return fig


