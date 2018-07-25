import argparse

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "-iterations",
						help = "Specify the number of minibatch iterations to train for",
						default = 1000000)
	parser.add_argument("-z", "-z_dim",
						help = "Specify the dimension of noise vector",
						default = 100)
	parser.add_argument("-mb", "-mb_size",
						help = "Size of a single minibatch",
						default = 64)
	parser.add_argument("-d", "-data_dir",
						help = "Dataset directory location",
						default = "data/faces")
	parser.add_argument("-di", "-disc_iter",
						help = "Number of iterations to train discriminator for per generator iter",
						default = 1)
	parser.add_argument("-s", "-save_interval",
						help = "Save the model every given iterations",
						default = 500)
	parser.add_argument("-m", "-multiplier",
						help = "Scale multiplier for model size, adjust so model fits on GPU",
						default = .25)
	parser.add_argument("-id", "-image_dim",
						help = "Dimension of the image (N x N)",
						default = 96)
	parser.add_argument("-p", "-print_interval",
						help = "Print log details and output image every so many iterations",
						default = 100)
	parser.add_argument("-l", "-loss_funciton",
						help = """Specify loss function from the following: 
									wass (wasserstein)
									drag (DRAGAN)
									rel  (Relativistic GAN)
									def  (Original default GAN)
									cond (Conditional GAN)
									""",
						default = "cond")
	parser.add_argument("-lm", "-load_model",
						help = """Specify location of model to load, assumes models are located
								  in folder \'Training-Models\' in cur directory. Provide meta 
								  file name in this directory""",
						default = None)
	parser.add_argument("-dlr", "-decay_learning_rate",
						help = "Causes the model to use a decaying learning rate",
						action = "store_true")
	parser.add_argument("-cc", "-coord_conv",
						help = "Uses coord conv layer instead of regular conv for first layer of disc",
						action = "store_true")
	parser.add_argument("-tp", "-tag_path",
						help = "Specify location of tags file for images",
						default = "data/tags_clean.csv")
	parser.add_argument("-ni", "-num_images",
						help = "Only load a set number of images",
						default = 30000)

	return parser.parse_args()


