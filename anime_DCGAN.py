from scipy import misc
import glob
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import prettytensor as pt
import os

def plot(samples):
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
		plt.imshow(sample.reshape(96, 96, 3), cmap='Greys_r')

	return fig

def load_images(path):
	image_list = []
	for image_path in glob.glob(path + "/*.png"):
		image = misc.imread(image_path)
		image = image.flatten()/127.5 - 1
		image_list.append(image)

	for image_path in glob.glob(path + "/*.jpg"):
		image = misc.imread(image_path)
		image = image.flatten()/127.5 - 1
		image_list.append(image)
	
	return image_list


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

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

def discriminator(x, isTrain = False, reuse = False):
	
	with tf.variable_scope('Discriminator', reuse = tf.AUTO_REUSE):
		conv1 = tf.layers.conv2d(x, 64, [5, 5], strides = (2,2), padding = 'same')
		lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training = isTrain))
		
		conv2 = tf.layers.conv2d(lrelu1, 128, [5, 5], strides = (2,2), padding = 'same')
		lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training = isTrain))

		conv3 = tf.layers.conv2d(lrelu2, 256, [5, 5], strides = (2,2), padding = 'same')
		lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training = isTrain))

		conv4 = tf.layers.conv2d(lrelu3, 512, [5, 5], strides = (2,2), padding = 'same')
		lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training = isTrain))
		lrelu4_flat = tf.reshape(lrelu3, [-1, 6*6*512])

		minibatch_disc = minibatch_discrimination(lrelu4_flat, 100)

		fc1 = tf.layers.dense(minibatch_disc, 1, activation = lrelu)
		D_prob = tf.nn.sigmoid(fc1)

	return D_prob, fc1

def generator(z, isTrain = True, reuse = False):

	with tf.variable_scope('Generator', reuse = reuse):

		fc0 = tf.layers.dense(z, 256*6*6, activation = lrelu)
		lrelu0 = lrelu(tf.layers.batch_normalization(fc0, training = isTrain))
		lrelu0 = tf.reshape(lrelu0, shape=[-1, 6, 6, 256])

		conv1 = tf.layers.conv2d_transpose(lrelu0, 512, [5, 5], strides = (2,2), padding = 'same')
		lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training = isTrain))

		conv2 = tf.layers.conv2d_transpose(lrelu1, 256, [5, 5], strides = (2,2), padding = 'same')
		lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training = isTrain))

		conv3 = tf.layers.conv2d_transpose(lrelu2, 128, [5, 5], strides = (2,2), padding = 'same')
		lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training = isTrain))

		conv4 = tf.layers.conv2d_transpose(lrelu3, 3, [5, 5], strides = (2,2), padding = 'same')
		output = tf.nn.tanh(conv4)

	return output

images = load_images('/images')


hist_averages = 5

X = tf.placeholder(tf.float32, shape=[None, 27648])
x_image = tf.reshape(X, [-1, 96, 96, 3])

Z = tf.placeholder(tf.float32, shape=[None, 1, 1, 100])

isTrain = tf.placeholder(dtype = tf.bool)


G_sample = generator(Z, isTrain)
D_real, D_logit_real = discriminator(x_image, isTrain)
D_fake, D_logit_fake = discriminator(G_sample, isTrain, reuse = True)

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')

#Default GAN 
"""
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_real, labels = tf.ones_like(D_logit_real)) - 0.1)
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.zeros_like(D_logit_fake)) + 0.1)

D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.ones_like(D_logit_fake)))
"""

#Wasterstein GAN
D_loss = tf.reduce_mean(D_logit_real) - tf.reduce_mean(D_logit_fake)
G_loss = -tf.reduce_mean(D_logit_fake)

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in dis_vars]


D_solver = tf.train.RMSPropOptimizer(learning_rate = 5e-4).minimize(-D_loss, var_list = dis_vars)
G_solver = tf.train.RMSPropOptimizer(learning_rate = 5e-4).minimize(G_loss, var_list = gen_vars)

if not os.path.exists('out/'):
    os.makedirs('out/')

mb_size = 64
Z_dim = 100
i = 0
n = 0
d_average = 0
g_average = 0

start_time = time.time()

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	#saver = tf.train.import_meta_graph('./Training Model/-10000.meta')
	#saver.restore(sess,tf.train.latest_checkpoint('Training Model'))

	for it in range(1000000):

		if it % 100 == 0:
			samples = sess.run(G_sample, feed_dict={Z: np.random.normal(0, 1, (16, 1, 1, Z_dim)), isTrain: True})
			fig = plot(samples)
			plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight')
			i += 1
			plt.close(fig)

		for _ in range(5):
			X_mb = images[n*mb_size:(n+1)*mb_size]
			n = (n + 1) % (len(images) // mb_size)
			_, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D], feed_dict = {X: X_mb, Z: np.random.normal(0, 1, (mb_size, 1, 1, Z_dim)), isTrain: True})
		
		_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {Z: np.random.normal(0, 1, (mb_size, 1, 1, Z_dim)), isTrain: True})
		
		d_average += D_loss_curr
		g_average += G_loss_curr

		if it % 100 == 0:
			d_average = d_average / 500
			g_average = g_average / 100
			print('Iter: {}'.format(it))
			print('D_loss: {:.4}'. format(d_average))
			print('G_loss: {:.4}'.format(g_average))
			d_average = 0
			g_average = 0

			end_time = time.time()
			print("Time taken: " + str(end_time - start_time))
			start_time = end_time
			print()
			
			if (it % 10000 == 0):
				saver.save(sess, 'Training Model/train_model', global_step = it)

	saver.save(sess, 'Final Model/Final_model')
