from scipy import misc
import glob
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
from ops import *

class model(object):

	def __init__(self):
		self.Z_dim = 100 #Noise vector dimensions
		self.IMAGE_DIM = 64 #output image size is IMAGE_DIM x IMAGE_DIM
		self.mb_size = 64 #Minibatch size
		self.print_interval = 10 #How often we print progress
		self.n = 0 #Minibatch seed
		self.images = self.load_images('data/faces') #Image data
		self.disc_iterations = 15 #Number of iterations to train disc for per gen iteration
		self.save_interval = 1000 #Save model every save_interval epochs
		self.max_iterations = 1000000 #Max iterations to train for

		#create output directory
		if not os.path.exists('out/'):
   			os.makedirs('out/')

		
	def __call__(self):
		self.build_model()
		self.train_model()


	#Loads images dataset normalizing and resizing them
	def load_images(self, path):
		image_list = []
		for i, image_path in enumerate(glob.glob(path + "/*")):
			image = misc.imread(image_path)
			image = misc.imresize(image, [self.IMAGE_DIM, self.IMAGE_DIM])
			image = image.flatten()/127.5 - 1
			image_list.append(image)
		
		return image_list


	#Builds the model to be run
	def build_model(self):

		#Image input tensors
		self.X = tf.placeholder(tf.float32, shape = [None, self.IMAGE_DIM * self.IMAGE_DIM*3])
		self.x_image = tf.reshape(self.X, [-1, self.IMAGE_DIM, self.IMAGE_DIM, 3])

		#Noise tensor
		self.Z = tf.placeholder(tf.float32, shape = [None, self.Z_dim])

		#Training tensor
		self.isTrain = tf.placeholder(dtype = tf.bool)

		self.wasserstein_GAN_loss()
		self.build_solver()


	#Wasserstein GAN loss function 
	def wasserstein_GAN_loss(self):

		#Get sample from generator
		self.G_sample = self.generator(self.Z, self.isTrain)

		#Get real and fake discriminator outputs
		D_logit_real = self.discriminator(self.x_image, self.isTrain)
		D_logit_fake = self.discriminator(self.G_sample, self.isTrain, reuse = True)

		#Improved WGAN https://arxiv.org/abs/1704.00028
		scale = 10
		epsilon = tf.random_uniform([], 0.0, 1.0)
		x_hat = epsilon * self.x_image + (1 - epsilon) * self.G_sample
		d_hat = self.discriminator(x_hat, self.isTrain, reuse = True)
		ddx = tf.gradients(d_hat, x_hat)[0]
		ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis = 1))
		ddx = tf.reduce_mean(tf.square(ddx - 1.0)) * scale

		self.D_loss = tf.reduce_mean(D_logit_real) - tf.reduce_mean(D_logit_fake) + ddx
		self.G_loss = tf.reduce_mean(D_logit_fake)
	

	#Vanilla GAN model loss function
	def vanilla_GAN_loss():

		#Get sample from generator
		self.G_sample = self.generator(self.Z, self.isTrain)

		#Get real and fake discriminator outputs
		D_logit_real = self.discriminator(self.x_image, self.isTrain)
		D_logit_fake = self.discriminator(G_sample, self.isTrain, reuse = True)

		D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_real, labels = tf.ones_like(D_logit_real)) - 0.1)
		D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.zeros_like(D_logit_fake)) + 0.1)

		self.D_loss = D_loss_real + D_loss_fake
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.ones_like(D_logit_fake)))


	#Builds the solver for graph, to be called after calling a loss function
	def build_solver(self):
		self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		self.dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')

		self.D_solver = tf.train.AdamOptimizer(learning_rate = 1e-4, beta1 = 0.5, beta2 = 0.9).minimize(self.D_loss, var_list = self.dis_vars)
		self.G_solver = tf.train.AdamOptimizer(learning_rate = 1e-4, beta1 = 0.5, beta2 = 0.9).minimize(self.G_loss, var_list = self.gen_vars)


	#Repeated block to be used in the discriminator
	def discriminator_block(self, inputs, out, kernel_size, stride):
	    res = inputs
	    net = tf.layers.conv2d(inputs, out, kernel_size, stride, padding = 'same', activation = None)
	    net = lrelu(net)
	    net = tf.layers.conv2d(net, out, kernel_size, stride, padding = 'same', activation = None)
	    net = net + inputs
	    net = lrelu(net)
	    return net


    #Creates the discriminator network
	def discriminator(self, x, isTrain = False, reuse = False):

		with tf.variable_scope('Discriminator', reuse = tf.AUTO_REUSE):
			net = tf.layers.conv2d(x, 32, 3, 1, padding = 'same')
			net = lrelu(net)

			net = self.discriminator_block(net, 32, 3, 1)

			net = tf.layers.conv2d(net, 64, 4, 2, padding = 'same')
			net = lrelu(net)

			net = self.discriminator_block(net, 64, 3, 1)

			net = tf.layers.conv2d(net, 128, 4, 2, padding = 'same')
			net = lrelu(net)

			net = self.discriminator_block(net, 128, 3, 1)

			net = tf.layers.conv2d(net, 256, 3, 2, padding = 'same')
			net = lrelu(net)

			net = self.discriminator_block(net, 256, 3, 1)

			net = tf.layers.conv2d(net, 512, 3, 2, padding = 'same')
			net = lrelu(net)

			net = self.discriminator_block(net, 512, 3, 1)

			net = tf.layers.conv2d(net, 512, 3, 2, padding = 'same')
			net = lrelu(net)

			logits = tf.layers.dense(net, 1)
			net = tf.nn.sigmoid(logits)

		return logits
	

	#Repeated block to be used in the generator
	def generator_block(self, inputs, out, kernel_size, stride, isTrain = True):

		net = tf.layers.conv2d_transpose(inputs, out, kernel_size, stride, padding = 'same')
		net = tf.layers.batch_normalization(net, training = isTrain)
		net = tf.nn.relu(net)
		net = tf.layers.conv2d_transpose(inputs, out, kernel_size, stride, padding = 'same')
		net = tf.layers.batch_normalization(net, training = isTrain)
		net = net + inputs

		return net


	#Creates the generator network
	def generator(self, z, isTrain = True, reuse = False):

		with tf.variable_scope('Generator', reuse = reuse):
			net = tf.layers.dense(z, 64*16*16)
			net = tf.layers.batch_normalization(net, training = isTrain)
			net = tf.reshape(net, [-1, 16, 16, 64])
			net = tf.nn.relu(net)

			inputs = net

			for i in range(4):
				net = self.generator_block(net, 64, 3, 1, isTrain)

			net = tf.layers.batch_normalization(net, training = isTrain)
			net = tf.nn.relu(net)
			net = inputs + net

			net = tf.layers.conv2d_transpose(net, 256, 3, 2, padding = 'same')
			net = tf.layers.batch_normalization(net, training = isTrain)
			net = tf.nn.relu(net)

			net = tf.layers.conv2d_transpose(net, 256, 3, 2, padding = 'same')
			net = tf.layers.batch_normalization(net, training = isTrain)
			net = tf.nn.relu(net)

			net = tf.layers.conv2d_transpose(net, 3, 3, 1, padding = 'same')

			output = tf.nn.tanh(net)

		return output	


	#Returns next batch of images
	def next_batch(self):
		mb = self.images[self.n*self.mb_size:(self.n+1)*self.mb_size]
		self.n = (self.n + 1) % (len(self.images) // self.mb_size)
		return mb


	#Returns random noise vector
	def noise_vec(self, size):
 		return np.random.normal(0, 1, (size, self.Z_dim))


	#Trains the model
	def train_model(self):
		start_time = time.time()

		with tf.Session() as sess:
			
			sess.run(tf.global_variables_initializer())

			self.saver = tf.train.Saver()
			
			d_average = 0 #Average disc loss
			g_average = 0 #Average gen loss

			for it in range(self.max_iterations):
				
				#Train discriminator
				for i in range(self.disc_iterations):
					feed = {self.X: self.next_batch(), self.Z: self.noise_vec(self.mb_size), self.isTrain: True}
					_, D_loss_curr = sess.run([self.D_solver, self.D_loss], feed_dict = feed)

				#Train generator
				feed = {self.Z: self.noise_vec(self.mb_size), self.isTrain: True}
				_, G_loss_curr = sess.run([self.G_solver, self.G_loss], feed_dict = feed)					
				
				d_average += D_loss_curr
				g_average += G_loss_curr
				
				#Store some progress images 
				if it % self.print_interval == 0:
					samples = sess.run(self.G_sample, feed_dict={self.Z: self.noise_vec(16), self.isTrain: True})
					fig = plot(samples, self.IMAGE_DIM)
					plt.savefig('out/{}.png'.format(str(it / self.print_interval).zfill(3)), bbox_inches = 'tight')
					plt.close(fig)
				
				#Print run time statistics
				if it % self.print_interval == 0:
					d_average = d_average / self.print_interval
					g_average = g_average / self.print_interval
					print('Iter: {}'.format(it))
					print('D_loss: {:.4}'. format(d_average))
					print('G_loss: {:.4}'.format(g_average))
					
					d_average = 0
					g_average = 0

					end_time = time.time()
					print("Time taken: " + str(end_time - start_time))
					start_time = end_time
					print()
					
				if (it % self.save_interval == 0):
					self.saver.save(sess, 'Training Model/train_model', global_step = it)

			self.saver.save(sess, 'Final Model/Final_model')


def main():
	GAN = model()
	GAN()


if __name__ == '__main__'
	main()

