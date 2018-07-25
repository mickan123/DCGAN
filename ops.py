import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import glob
import os
import time
from scipy import misc
from random import shuffle

#Leaky relu function
def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

#Implements minibatch discrimination https://arxiv.org/abs/1606.03498
def minibatch_discrimination(input_layer, num_kernels, dim_per_kernel = 3):

	input_layer = tf.layers.Flatten()(input_layer)
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


hair_colour = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
	'green hair', 'red hair', 'purple hair', 'pink hair',
	'blue hair', 'black hair', 'brown hair', 'blonde hair']

eye_colour = ['gray eyes', 'black eyes', 'orange eyes',
	'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
	'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

def make_one_hot(tags, eye_map, hair_map):
	eye = []
	hair = []
	for tag in tags:
		if tag in hair_map:
			hair.append(hair_map[tag])
		if tag in eye_map:
			eye.append(eye_map[tag])

	eyes_hot = np.zeros([len(eye_colour)])
	hair_hot = np.zeros([len(hair_colour)])

	for idx in eye:
		eyes_hot[idx] = 1
	for idx in hair:
		hair_hot[idx] = 1

	tag_vec = np.concatenate((eyes_hot, hair_hot))

	return tag_vec

#Generates tags from unclean tag file and saves in one hot format
def generate_tags(tag_path):
	tags = pd.read_csv(tag_path, header = None)
	tag_values = tags[1].tolist()
	tag_values = [''.join([i for i in s if not i.isdigit()]) for s in tag_values] #Remove digits form strings
	tag_values = [s.replace('\t', "") for s in tag_values]                        #Remove \t
	tag_values = [s.split(':') for s in tag_values]                               #Split by separator

	hair_map = {}
	eye_map = {}
	for idx, h in enumerate(hair_colour):
		hair_map[h] = idx
	for idx, e in enumerate(eye_colour):
		eye_map[e] = idx

	one_hot_tags = []
	for tags in tag_values:
		one_hot_tags.append(make_one_hot(tags, eye_map, hair_map))

	labels = eye_colour + hair_colour
	df = pd.DataFrame(np.asarray(one_hot_tags), columns = labels)
	df.to_csv("data/tags_one_hot.csv", index=False)

#Loads images dataset normalizing and resizing them
def load_images(size, path, num_images = 30000):
	image_list = []
	for i in range(num_images):
		image_path = path + "/" + str(i) + ".jpg"
		image = misc.imread(image_path)
		image = misc.imresize(image, [size, size])
		image = image.flatten()/127.5 - 1
		image_list.append(image)

	tags_df = pd.read_csv("data/tags_one_hot.csv", index_col=False)
	trimmed_tag_list = tags_df.values.tolist()[:num_images] #Get first num_images
	full_df = pd.DataFrame(np.asarray(trimmed_tag_list))

	#Shuffle tags and images together
	full_df['values'] = image_list #Add image data to list
	image_data = full_df.values.tolist()
	shuffle(image_data)

	#Extract images and tags
	num_tags = len(hair_colour) + len(eye_colour)
	tags = [data[:num_tags] for data in image_data]
	imgs = [data[num_tags] for data in image_data]
	return imgs, tags

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
		plt.imshow(sample, cmap='Greys_r')

	return fig

#Save a single image to output folder
def save_image(image, dim, image_name, output_dir = 'images/'):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	fig = plt.figure()
	plt.axis('off')
	image = np.asarray(np.reshape(image, [dim,dim,3]))
	image = (image + 1) / 2

	plt.imshow(image, cmap='Greys_r')
	plt.savefig(output_dir + image_name)
	plt.close(fig)


def pixelShuffler(inputs, scale = 2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output