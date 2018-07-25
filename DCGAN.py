
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
from ops import *
from model import *
from arguments import parse_arguments

class model(object):

	def __init__(self, args):
		
		self.IMAGE_DIM = int(args.id) #output image size is IMAGE_DIM x IMAGE_DIM
		if (args.tp != None):
			generate_tags(args.tp)
		self.images, self.tags = load_images(self.IMAGE_DIM, args.d, num_images = int(args.ni)) #Image dataset
		self.tag_len = len(self.tags[0]) #Number of tags
		self.eye_len = 11 #Number of eye colours
		self.hair_len = 12 #Number of hair colours

		self.disc_iterations = int(args.di) #Number of iterations to train disc for per gen iteration
		self.max_iterations = int(args.i) #Max iterations to train for
		self.save_interval = int(args.s) #Save model every save_interval epochs
		self.print_interval = int(args.p) #How often we print progress
		
		self.mb_size = int(args.mb) #Minibatch size
		self.Z_dim = int(args.z) #Noise vector dimensions
		self.mult = float(args.m) #Scalar multiplier for model size
		self.loss = args.l #Loss function to use
		self.decay_lr = args.dlr #Boolean on whether to use decaying learning rate
		self.coord_conv = args.cc #Boolean on whether to use coord conv
		
		self.lambd = .5 #Used for DRAGAN
		self.loss_label_mul = 5 #Weighting for label vs tag for cond GAN
		self.learning_rate = 2e-4
		self.n = 0 #Minibatch seed
		self.it = 0 #Current iteration

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
		self.real_tag = tf.placeholder(tf.float32, [None, self.tag_len], name="real_tag")

		#Noise tensor
		self.Z = tf.placeholder(tf.float32, shape = [None, self.Z_dim])
		self.fake_tag = tf.placeholder(tf.float32, [None, self.tag_len], name="rand_seq")

		#Training tensor
		self.isTrain = tf.placeholder(dtype = tf.bool)

		#Create global step value
		tf.train.create_global_step()

		#Create loss function
		if (self.loss == "wass"):
			self.wasserstein_GAN_loss()
		elif (self.loss == "drag"):
			self.DRAGAN_loss()
		elif (self.loss == "rel"):
			self.relativistic_GAN_loss()
		elif (self.loss == "cond"):
			self.conditional_GAN_loss()
		else:
			self.vanilla_GAN_loss()

		self.build_solver()

	#Wasserstein GAN loss function 
	def wasserstein_GAN_loss(self):

		#Get sample from generator
		self.G_sample = generator(self.Z, self.mult, isTrain = self.isTrain)

		#Get real and fake discriminator outputs
		D_logit_real, _ = discriminator(self.X, self.IMAGE_DIM, self.mult, self.isTrain)
		D_logit_fake, _ = discriminator(self.G_sample, self.IMAGE_DIM, self.mult, self.isTrain, reuse = True)

		#Improved WGAN https://arxiv.org/abs/1704.00028
		scale = 10
		epsilon = tf.random_uniform([], 0.0, 1.0)
		x_hat = epsilon * self.X + (1 - epsilon) * tf.reshape(self.G_sample, [-1, self.IMAGE_DIM * self.IMAGE_DIM * 3])
		d_hat = discriminator(x_hat, self.IMAGE_DIM, self.mult, self.isTrain, reuse = True)
		ddx = tf.gradients(d_hat, x_hat)[0]
		ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis = 1))
		ddx = tf.reduce_mean(tf.square(ddx - 1.0)) * scale

		self.D_loss = tf.reduce_mean(D_logit_real) - tf.reduce_mean(D_logit_fake) + ddx
		self.G_loss = tf.reduce_mean(D_logit_fake)

	#From paper https://arxiv.org/pdf/1705.07215.pdf
	def DRAGAN_loss(self):

		#Get sample from generator
		self.G_sample = generator(self.Z, self.mult, isTrain = self.isTrain)

		#Get real and fake discriminator outputs
		D_logit_real, _ = discriminator(self.X,self.IMAGE_DIM, self.mult, self.isTrain)
		D_logit_fake, _ = discriminator(self.G_sample, self.IMAGE_DIM, self.mult, self.isTrain, reuse = True)

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
		gradients = tf.gradients(discriminator(interpolates, self.IMAGE_DIM, self.mult, self.isTrain), [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
		self.D_loss += self.lambd * gradient_penalty


	#From paper https://arxiv.org/pdf/1807.00734.pdf
	def relativistic_GAN_loss(self):

		#Get sample from generator
		self.G_sample = generator(self.Z, self.mult, isTrain = self.isTrain)

		#Get real and fake discriminator outputs
		D_logit_real, _ = discriminator(self.X, self.IMAGE_DIM, self.mult, self.isTrain)
		D_logit_fake, _ = discriminator(self.G_sample, self.IMAGE_DIM, self.mult, self.isTrain, reuse = True)

		self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_real - D_logit_fake, labels = tf.ones_like(D_logit_real)))
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake - D_logit_real, labels = tf.ones_like(D_logit_real)))

	#Implements vanilla GAN loss with conditional loss from tags
	def conditional_GAN_loss(self):

		#Get fake sample from generator
		self.G_sample = generator(self.Z, self.mult, self.fake_tag, self.isTrain)

		#Get real and fake discriminator outputs
		pred_real, pred_real_tag = discriminator(self.X, self.IMAGE_DIM, self.mult, self.isTrain)
		pred_fake, pred_fake_tag = discriminator(self.G_sample, self.IMAGE_DIM, self.mult, self.isTrain, reuse = True)

		loss_real_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = pred_real, labels = tf.ones_like(pred_real)))
		loss_real_tag = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = pred_real_tag, labels = self.real_tag), axis = 1))
		loss_d_real = 23 * loss_real_label + loss_real_tag

		loss_fake_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = pred_fake, labels = tf.zeros_like(pred_fake)))
		loss_fake_tag = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = pred_fake_tag, labels = self.fake_tag), axis = 1))
		loss_d_fake = 23 * loss_fake_label + loss_fake_tag

		self.D_loss = loss_d_real + loss_d_fake

		loss_g_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = pred_fake, labels = tf.ones_like(pred_fake)))
		loss_g_tag = loss_fake_tag

		self.G_loss = 23 * loss_g_label + loss_g_tag

		#Tag loss for debugging
		self.G_tag_loss = loss_g_tag
		self.D_tag_loss = loss_real_tag + loss_fake_tag


	#Vanilla GAN model loss function
	def vanilla_GAN_loss(self):

		#Get sample from generator
		self.G_sample = generator(self.Z, self.mult, isTrain = self.isTrain)

		#Get real and fake discriminator outputs
		D_logit_real, _ = discriminator(self.X, self.IMAGE_DIM, self.mult, self.isTrain)
		D_logit_fake, _ = discriminator(self.G_sample, self.IMAGE_DIM, self.mult, self.isTrain, reuse = True)

		D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_real, labels = tf.ones_like(D_logit_real)))
		D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.zeros_like(D_logit_fake)))

		self.D_loss = D_loss_real + D_loss_fake
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.ones_like(D_logit_fake)))

	#Builds the solver for graph, to be called after calling a loss function
	def build_solver(self):
		self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		self.dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')

		if (self.decay_lr):
			self.learning_rate = tf.train.exponential_decay(
								  2e-4,                        # Base learning rate.
								  tf.train.get_global_step(),  # Current index into the dataset.
								  self.mb_size,                # Decay step.
								  0.995,                       # Decay rate.
								  staircase=True)
		else:
			self.learning_rate = 2e-4
		
		self.D_solver = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = 0.5, beta2 = 0.999).minimize(self.D_loss, var_list = self.dis_vars)
		self.G_solver = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = 0.5, beta2 = 0.999).minimize(self.G_loss, var_list = self.gen_vars, global_step = tf.train.get_global_step())

	#Returns next batch of images
	def next_batch(self):
		mb_imgs = self.images[self.n*self.mb_size:(self.n+1)*self.mb_size]
		mb_tags = self.tags[self.n*self.mb_size:(self.n+1)*self.mb_size]
		self.n = (self.n + 1) % (len(self.images) // self.mb_size)
		return mb_imgs, mb_tags

	def get_perturbed_batch(self, minibatch):
		minibatch = np.asarray(minibatch)
		return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)

	#Returns random noise vector
	def noise_vec(self, size):
		return np.random.normal(0, 1, (size, self.Z_dim))

	#Generates a fake tag with 1 hair colours and 1 eye colour
	def gen_fake_tag(self, batch_size):
		eye_label = np.random.randint(0, self.eye_len, batch_size)
		hair_label = np.random.randint(0, self.hair_len, batch_size)
		random_tag = np.zeros((batch_size, self.tag_len))
		random_tag[np.arange(batch_size), eye_label] = 1
		random_tag[np.arange(batch_size), hair_label + self.eye_len] = 1
		return random_tag

	#Train step for the discriminator
	def discriminator_train_step(self):
		total_loss = 0

		for i in range(self.disc_iterations):
			imgs, tags = self.next_batch()
			feed = {self.X: imgs, 
					self.real_tag: tags,
					self.Z: self.noise_vec(self.mb_size), 
					self.fake_tag: self.gen_fake_tag(self.mb_size),
					self.isTrain: True}

			if (self.loss == "drag"):
				feed[self.X_p] = self.get_perturbed_batch(imgs)

			_, loss = self.sess.run([self.D_solver, self.D_loss], feed_dict = feed)
			total_loss += loss

		return total_loss

	def generator_train_step(self):
		feed = {self.Z: self.noise_vec(self.mb_size), 
				self.fake_tag: self.gen_fake_tag(self.mb_size),
				self.isTrain: True}

		if (self.loss == "rel"):
			imgs, tags = self.next_batch()
			feed[self.X] = imgs

		_, loss = self.sess.run([self.G_solver, self.G_loss], feed_dict = feed)
		return loss

	def generate_statistics(self, d_loss, g_loss):
		#Generate and save some samples to out/ folder
		feed = {self.Z: self.noise_vec(16), 
				self.fake_tag: self.gen_fake_tag(16),
				self.isTrain: False}
		samples = self.sess.run(self.G_sample, feed_dict = feed)
		fig = plot(samples, self.IMAGE_DIM)
		plt.savefig('out/{}.png'.format(str(self.it / self.print_interval).zfill(3)), bbox_inches = 'tight')
		plt.close(fig)

		#Print some model statistics to stdout
		d_loss = d_loss / (self.print_interval * self.disc_iterations)
		g_loss = g_loss / self.print_interval
		print('Iter: {}'.format(self.it))
		print('Disc loss: {:.4}'. format(d_loss))
		print('Gen loss: {:.4}'.format(g_loss))

		self.end_time = time.time()
		print("Time taken: " + str(self.end_time - self.start_time))
		self.start_time = self.end_time
		print()

	#Trains the model
	def train_model(self):
		self.start_time = time.time()

		with tf.Session() as self.sess:

			self.sess.run(tf.global_variables_initializer())

			self.saver = tf.train.Saver()
			
			if (self.load_model != None):
				self.saver = tf.train.import_meta_graph('Training Model/' + self.load_model)
				self.aver.restore(self.sess, tf.train.latest_checkpoint('Training Model/'))

			gen_avg_loss = 0
			disc_avg_loss = 0

			for self.it in range(1, self.max_iterations):
				disc_avg_loss += self.discriminator_train_step()
				gen_avg_loss += self.generator_train_step()
				
				#Store some progress images 
				if self.it % self.print_interval == 0:
					self.generate_statistics(disc_avg_loss, gen_avg_loss)
					disc_avg_loss = 0
					gen_avg_loss = 0
					
				if (self.it % self.save_interval == 0):
					self.saver.save(self.sess, 'Training Model/train_model', global_step = self.it)

			self.saver.save(self.sess, 'Final Model/Final_model')


def main():
	args = parse_arguments()
	GAN = model(args)
	GAN()

if __name__ == '__main__':
	main()

