
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
from ops import *
from CoordConv import CoordConv
from arguments import parse_arguments

class model(object):

	def __init__(self, args):
		
		self.IMAGE_DIM = int(args.id) #output image size is IMAGE_DIM x IMAGE_DIM
		#self.images = load_images(self.IMAGE_DIM, args.d) #Image dataset

		self.disc_iterations = int(args.di) #Number of iterations to train disc for per gen iteration
		self.max_iterations = int(args.i) #Max iterations to train for
		self.save_interval = int(args.s) #Save model every save_interval epochs
		self.print_interval = int(args.p) #How often we print progress
		
		self.mb_size = int(args.mb) #Minibatch size
		self.Z_dim = int(args.z) #Noise vector dimensions
		self.mult = float(args.m) #Scalar multiplier for model size
		self.loss = args.l #Loss function to use
		
		self.lambd = .5 #Used for DRAGAN
		self.n = 0 #Minibatch seed

		self.load_model = args.lm #Model to load

		#create output directory
		if not os.path.exists('out/'):
   			os.makedirs('out/')

	def __call__(self):
		self.build_model()
		self.train_model()

	#Builds the model to be run
	def build_model(self):

		#Image input tensors
		self.X = tf.placeholder(tf.float32, shape = [None, self.IMAGE_DIM * self.IMAGE_DIM * 3])
		self.X_p = tf.placeholder(tf.float32, shape = [None, self.IMAGE_DIM * self.IMAGE_DIM * 3]) #For DRAGAN

		#Noise tensor
		self.Z = tf.placeholder(tf.float32, shape = [None, self.Z_dim])

		#Training tensor
		self.isTrain = tf.placeholder(dtype = tf.bool)

		#Create loss function
		if (self.loss == "wass"):
			self.wasserstein_GAN_loss()
		elif (self.loss == "drag"):
			self.DRAGAN_loss()
		elif (self.loss == "rel"):
			self.relativistic_GAN_loss()
		else:
			self.vanilla_GAN_loss()

		self.build_solver()

	#Wasserstein GAN loss function 
	def wasserstein_GAN_loss(self):

		#Get sample from generator
		self.G_sample = self.generator(self.Z, self.isTrain)

		#Get real and fake discriminator outputs
		D_logit_real = self.discriminator(self.X, self.isTrain)
		D_logit_fake = self.discriminator(self.G_sample, self.isTrain, reuse = True)

		#Improved WGAN https://arxiv.org/abs/1704.00028
		scale = 10
		epsilon = tf.random_uniform([], 0.0, 1.0)
		x_hat = epsilon * self.X + (1 - epsilon) * self.G_sample
		d_hat = self.discriminator(x_hat, self.isTrain, reuse = True)
		ddx = tf.gradients(d_hat, x_hat)[0]
		ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis = 1))
		ddx = tf.reduce_mean(tf.square(ddx - 1.0)) * scale

		self.D_loss = tf.reduce_mean(D_logit_real) - tf.reduce_mean(D_logit_fake) + ddx
		self.G_loss = tf.reduce_mean(D_logit_fake)

	#From paper https://arxiv.org/pdf/1705.07215.pdf
	def DRAGAN_loss(self):

		#Get sample from generator
		self.G_sample = self.generator(self.Z, self.isTrain)

		#Get real and fake discriminator outputs
		D_logit_real = self.discriminator(self.X, self.isTrain)
		D_logit_fake = self.discriminator(self.G_sample, self.isTrain, reuse = True)

		D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_real, labels = tf.ones_like(D_logit_real)))
		D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.zeros_like(D_logit_fake)))

		self.D_loss = D_loss_real + D_loss_fake
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.ones_like(D_logit_fake)))

		#Gradient penalty
		alpha = tf.random_uniform(
		    shape=[self.mb_size, 1], 
		    minval=0.,
		    maxval=1.
		)
		differences = self.X - self.X_p
		interpolates = self.X + (alpha * differences)
		gradients = tf.gradients(self.discriminator(interpolates, self.isTrain), [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
		self.D_loss += self.lambd * gradient_penalty


	#From paper https://arxiv.org/pdf/1807.00734.pdf
	def relativistic_GAN_loss(self):

		#Get sample from generator
		self.G_sample = self.generator(self.Z, self.isTrain)

		#Get real and fake discriminator outputs
		D_logit_real = self.discriminator(self.X, self.isTrain)
		D_logit_fake = self.discriminator(self.G_sample, self.isTrain, reuse = True)

		self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_real - D_logit_fake, labels = tf.ones_like(D_logit_real)))
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake - D_logit_real, labels = tf.ones_like(D_logit_real)))
	
	#Vanilla GAN model loss function
	def vanilla_GAN_loss():

		#Get sample from generator
		self.G_sample = self.generator(self.Z, self.isTrain)

		#Get real and fake discriminator outputs
		D_logit_real = self.discriminator(self.X, self.isTrain)
		D_logit_fake = self.discriminator(G_sample, self.isTrain, reuse = True)

		D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_real, labels = tf.ones_like(D_logit_real)))
		D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.zeros_like(D_logit_fake)))

		self.D_loss = D_loss_real + D_loss_fake
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.ones_like(D_logit_fake)))

	#Builds the solver for graph, to be called after calling a loss function
	def build_solver(self):
		self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		self.dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')

		self.D_solver = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5, beta2 = 0.999).minimize(self.D_loss, var_list = self.dis_vars)
		self.G_solver = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5, beta2 = 0.999).minimize(self.G_loss, var_list = self.gen_vars)

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

		x_image = tf.reshape(x, [-1, self.IMAGE_DIM, self.IMAGE_DIM, 3])

		with tf.variable_scope('Discriminator', reuse = tf.AUTO_REUSE):
			net = CoordConv(x_image, 32 * self.mult, 3, 1, padding = 'same')
			net = lrelu(net)

			net = self.discriminator_block(net, 32 * self.mult, 3, 1)

			net = tf.layers.conv2d(net, 64 * self.mult, 4, 2, padding = 'same')
			net = lrelu(net)

			for _ in range(4):
				net = self.discriminator_block(net, 64 * self.mult, 3, 1)

			net = tf.layers.conv2d(net, 128 * self.mult, 4, 2, padding = 'same')
			net = lrelu(net)

			for _ in range(4):
				net = self.discriminator_block(net, 128 * self.mult, 3, 1)

			net = tf.layers.conv2d(net, 256 * self.mult, 3, 2, padding = 'same')
			net = lrelu(net)

			for _ in range(4):
				net = self.discriminator_block(net, 256 * self.mult, 3, 1)

			net = tf.layers.conv2d(net, 512 * self.mult, 3, 2, padding = 'same')
			net = lrelu(net)

			for _ in range(4):
				net = self.discriminator_block(net, 512 * self.mult, 3, 1)

			net = tf.layers.conv2d(net, 512 * self.mult, 3, 2, padding = 'same')
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
			net = tf.layers.dense(z, 64*24*24 * self.mult)
			net = tf.layers.batch_normalization(net, training = isTrain)
			net = tf.reshape(net, [-1, 24, 24, int(64 * self.mult)])
			net = tf.nn.relu(net)

			inputs = net

			for i in range(16):
				net = self.generator_block(net, int(64 * self.mult), 3, 1, isTrain)

			net = tf.layers.batch_normalization(net, training = isTrain)
			net = tf.nn.relu(net)
			net = inputs + net

			net = tf.layers.conv2d_transpose(net, int(256 * self.mult), 3, 2, padding = 'same')
			net = tf.layers.batch_normalization(net, training = isTrain)
			net = tf.nn.relu(net)

			net = tf.layers.conv2d_transpose(net, int(256 * self.mult), 3, 2, padding = 'same')
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

	def get_perturbed_batch(self, minibatch):
		minibatch = np.asarray(minibatch)
		return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)

	#Returns random noise vector
	def noise_vec(self, size):
		return np.random.normal(0, 1, (size, self.Z_dim))

	#Train step for the discriminator
	def discriminator_train_step(self):
		total_loss = 0

		for i in range(self.disc_iterations):
			batch = self.next_batch()
			feed = {self.X: batch, self.Z: self.noise_vec(self.mb_size), self.isTrain: True}

			if (self.loss == "drag"):
				feed[self.X_p] = get_perturbed_batch(batch)

			_, loss = sess.run([self.D_solver, self.D_loss], feed_dict = feed)
			total_loss += loss

		return total_loss

	def generator_train_step(self):
		feed = {self.Z: self.noise_vec(self.mb_size), self.isTrain: True}

		if (self.loss == "rel"):
			feed[self.X] = self.next_batch()

		_, loss = sess.run([self.G_solver, self.G_loss], feed_dict = feed)

		return loss

	def generate_statistics(self, d_loss, g_loss):
		#Generate and save some samples to out/ folder
		samples = sess.run(self.G_sample, feed_dict={self.Z: self.noise_vec(16), self.isTrain: True})
		fig = plot(samples, self.IMAGE_DIM)
		plt.savefig('out/{}.png'.format(str(it / self.print_interval).zfill(3)), bbox_inches = 'tight')
		plt.close(fig)

		#Print some model statistics to stdout
		d_loss = d_loss / (self.print_interval * self.disc_iterations)
		g_loss = g_loss / self.print_interval
		print('Iter: {}'.format(it))
		print('D_loss: {:.4}'. format(d_loss))
		print('G_loss: {:.4}'.format(g_loss))

		end_time = time.time()
		print("Time taken: " + str(end_time - start_time))
		start_time = end_time
		print()

	#Trains the model
	def train_model(self):
		start_time = time.time()

		with tf.Session() as sess:
			
			sess.run(tf.global_variables_initializer())

			self.saver = tf.train.Saver()
			
			if (self.load_model != None):
				self.saver = tf.train.import_meta_graph('Training Model/' + self.load_model)
				self.aver.restore(sess, tf.train.latest_checkpoint('Training Model/'))

			disc_avg_loss = 0 
			gen_avg_loss = 0

			for it in range(1, self.max_iterations):
				
				disc_avg_loss += self.discriminator_train_step()
				gen_avg_loss += self.generator_train_step()
				
				#Store some progress images 
				if it % self.print_interval == 0:
					self.generate_statistics()
					disc_avg_loss = 0
					gen_avg_loss = 0
					
				if (it % self.save_interval == 0):
					self.saver.save(sess, 'Training Model/train_model', global_step = it)

			self.saver.save(sess, 'Final Model/Final_model')


def main():
	args = parse_arguments()
	GAN = model(args)
	GAN()

if __name__ == '__main__':
	main()

