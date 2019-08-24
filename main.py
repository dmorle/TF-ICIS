import os
import pickle
from GRU import *
from CNN import *
import matplotlib.pyplot as plt

dataPath = "../data/subsetData/"

def network(image, K, W, U, b, h, GRUStacks, M, N, subsetNum):

	# running the CNN
	imageConvRunData = CNN(image, K)

	# running the GRU
	for k in range(subsetNum):
		(h, GRUStacks) = GRU_Cell(M, N, h, imageConvRunData[0][5][k], W, U, b, GRUStacks)

	return (imageConvRunData, GRUStacks)

def backpropGRU(runData, s, ks, K, W, U, b, y, M, N, subsetNum, W_grad, U_grad, b_grad):

	# unpacking inputs
	(imageConvRunData, GRUStacks) = runData
	(Z_Stack, ZT_Stack, R_Stack, RT_Stack, P_Stack, PT_Stack, H_Stack) = GRUStacks
	(Wz_grad, Wr_grad, Wp_grad) = W_grad
	(Uz_grad, Ur_grad, Up_grad) = U_grad
	(bz_grad, br_grad, bp_grad) = b_grad

	# cross-entropy error between y and H_Stack[subsetNum-1] to get h_grad
	h_grad = tf.reshape(
		tf.multiply(
			tf.cast(-1/N, dtype = tf.float64),
			tf.add(
				tf.math.xdivy(
					y,
					tf.reshape(
						tf.add(
							tf.constant(1, dtype = tf.float64),
							H_Stack[subsetNum]
						),
						[N]
					)
				),
				tf.math.xdivy(
					tf.subtract(
						tf.constant(1, dtype = tf.float64),
						y
					),
					tf.reshape(
						tf.subtract(
							tf.constant(1, dtype = tf.float64),
							H_Stack[subsetNum]
						),
						[N]
					)
				)
			)
		),
		[N, 1]
	)

	# for CNN gradient
	xn_grad = tf.zeros(name = "xn_grad", shape = (0, 1, 1, 50), dtype = tf.float64)

	# calculating the gradient wrt GRU parameters
	for i in range(subsetNum-1, -1, -1):
		# TODO: handle the batch dimension given by CNN

		# unpacking CNN data
		(A, Z) = imageConvRunData
		(A0, A1, A2, A3, A4, A5) = A
		(Z0, Z2, Z4) = Z

		# preparing inputs for backprop
		internalStates = (Z_Stack[i], ZT_Stack[i], R_Stack[i], RT_Stack[i], P_Stack[i], PT_Stack[i])
		H = H_Stack[i]
		X = tf.reshape(A5[i], (M, 1))

		# getting GRU gradient
		(
			h_grad, xni_grad,
			(Wzi_grad, Wri_grad, Wpi_grad),
			(Uzi_grad, Uri_grad, Upi_grad),
			(bzi_grad, bri_grad, bpi_grad)
		) = backPropGRU_Cell(W, U, b, internalStates, H, X, h_grad, layer = i)

		xn_grad = tf.concat((tf.reshape(xni_grad, (1, 1, 1, 50)), xn_grad), 0)

		# updating GRU gradients
		Wz_grad = tf.add(Wz_grad, Wzi_grad)
		Wr_grad = tf.add(Wr_grad, Wri_grad)
		Wp_grad = tf.add(Wp_grad, Wpi_grad)
		Uz_grad = tf.add(Uz_grad, Uzi_grad)
		Ur_grad = tf.add(Ur_grad, Uri_grad)
		Up_grad = tf.add(Up_grad, Upi_grad)
		bz_grad = tf.add(bz_grad, bzi_grad)
		br_grad = tf.add(br_grad, bri_grad)
		bp_grad = tf.add(bp_grad, bpi_grad)

	W_grad = (Wz_grad, Wr_grad, Wp_grad)
	U_grad = (Uz_grad, Ur_grad, Up_grad)
	b_grad = (bz_grad, br_grad, bp_grad)

	return (xn_grad, W_grad, U_grad, b_grad)

def load_externals(batchNum, batchSize, Inputs, Labels):
	# number of inputs available per label
	dataSizes = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

	# getting the number of inputs needed for each label
	inputSizes = [batchSize//10]*10
	for i in range(batchSize%10):
		inputSizes[i] += 1

	# setting the initialization for all placeholders
	externals = dict()
	for k in range(batchNum):
		# generate new input numbers for each batch
		inputNums = list()
		for i in range(10):
			inputNums.append(np.random.randint(dataSizes[i], size = inputSizes[i]))

		print("\nfor batch " + str(k) + ", the input images are as follows:")
		for num in inputNums:
			print(num)
		print()

		# initialize the externals for batch k
		for i in range(len(inputNums)):
			for j in range(len(inputNums[i])):
				n = i + 10*j
				with open(dataPath + str(i) + "/Image_" + "{:04d}".format(inputNums[i][j]), "rb") as f:
					externals[Inputs[k][n]] = np.transpose(pickle.load(f), (0, 2, 3, 1))

				label = [0]*10
				label[i] = 1
				externals[Labels[k][n]] = label
		
	return externals

def getSaver(K, W, U, b):
	(K0, K2, K4) = K
	(Wz, Wr, Wp) = W
	(Uz, Ur, Up) = U
	(bz, br, bp) = b

	# dictionary of all network parameters
	saveDict = {
		"K0" : K0, "K2" : K2, "K4" : K4,
		"Wz" : Wz, "Wr" : Wr, "Wp" : Wp,
		"Uz" : Uz, "Ur" : Ur, "Up" : Up,
		"bz" : bz, "br" : br, "bp" : bp
	}

	return tf.train.Saver(saveDict)

def runTrainSession(run_op, saver, externals, save_path, load_path = None):

	if load_path:
		# running the graph
		with tf.Session() as sess:
			saver.restore(sess, load_path)

			writer = tf.summary.FileWriter('graphs', sess.graph)

			sess.run(run_op, feed_dict = externals)

			saver.save(sess, save_path)

	else:
		init_op = tf.global_variables_initializer()

		# running the graph
		with tf.Session() as sess:
			sess.run(init_op)

			writer = tf.summary.FileWriter('graphs', sess.graph)

			sess.run(run_op, feed_dict = externals)

			saver.save(sess, save_path)
	return

def trainNetwork(learningRate, batchNum, batchSize, savePath, loadPath = None):
	# resetting the current graph
	tf.reset_default_graph()

	# hyper-parameters

	subsetNum = 4	# Number of subsets per image
	subsetSize = (14, 14, 1)	# Size of each image subset

	M = 50	# dimension of feature map
	N = 10	# number of labels

	learningRate = tf.cast(learningRate, dtype = tf.float64)

	s0 = (subsetNum, 14, 14, 1)
	s1 = (subsetNum, 12, 12, 10)
	s2 = (subsetNum, 6, 6, 10)
	s3 = (subsetNum, 4, 4, 25)
	s4 = (subsetNum, 2, 2, 25)
	s5 = (subsetNum, 1, 1, 50)
	s = (s0, s1, s2, s3, s4, s5)

	ks0 = (3, 3, 1, 10)
	ks2 = (3, 3, 10, 25)
	ks4 = (2, 2, 25, 50)
	ks = (ks0, ks2, ks4)

	"""
		Creating the CNN variables

	"""

	K0 = tf.get_variable(name = "K0", shape = ks0, initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	K2 = tf.get_variable(name = "K2", shape = ks2, initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	K4 = tf.get_variable(name = "K4", shape = ks4, initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)

	"""
		Creating the GRU cell variables

	"""

	# W matricies are multiplied with h(t-1)
	Wz = tf.get_variable(name = "Wz", shape = (N, M), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	Wr = tf.get_variable(name = "Wr", shape = (N, M), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	Wp = tf.get_variable(name = "Wp", shape = (N, M), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)

	# U matricies are multiplied with x(t-1)
	Uz = tf.get_variable(name = "Uz", shape = (N, N), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	Ur = tf.get_variable(name = "Ur", shape = (N, N), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	Up = tf.get_variable(name = "Up", shape = (N, N), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	
	# b vectors are N dimensional biases
	bz = tf.get_variable(name = "bz", shape = (N, 1), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	br = tf.get_variable(name = "br", shape = (N, 1), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	bp = tf.get_variable(name = "bp", shape = (N, 1), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)

	# packing network parameters
	K = (K0, K2, K4)
	W = (Wz, Wr, Wp)
	U = (Uz, Ur, Up)
	b = (bz, br, bp)

	# state h starts as a constant of 0s
	h = tf.zeros(name = "h", shape = (N, 1), dtype = tf.float64)

	# initializing the GRU internal state stacks
	Z_Stack = list()
	ZT_Stack = list()
	R_Stack = list()
	RT_Stack = list()
	P_Stack = list()
	PT_Stack = list()
	H_Stack = [h]
	stacks = (Z_Stack, ZT_Stack, R_Stack, RT_Stack, P_Stack, PT_Stack, H_Stack)

	# initializing GRU gradients
	Wz_grad = tf.zeros(name = "Wz_grad", shape = (N, M), dtype = tf.float64)
	Wr_grad = tf.zeros(name = "Wr_grad", shape = (N, M), dtype = tf.float64)
	Wp_grad = tf.zeros(name = "Wp_grad", shape = (N, M), dtype = tf.float64)
	W_grad = (Wz_grad, Wr_grad, Wp_grad)

	Uz_grad = tf.zeros(name = "Uz_grad", shape = (N, N), dtype = tf.float64)
	Ur_grad = tf.zeros(name = "Ur_grad", shape = (N, N), dtype = tf.float64)
	Up_grad = tf.zeros(name = "Up_grad", shape = (N, N), dtype = tf.float64)
	U_grad = (Uz_grad, Ur_grad, Up_grad)

	bz_grad = tf.zeros(name = "bz_grad", shape = (N, 1), dtype = tf.float64)
	br_grad = tf.zeros(name = "br_grad", shape = (N, 1), dtype = tf.float64)
	bp_grad = tf.zeros(name = "bp_grad", shape = (N, 1), dtype = tf.float64)
	b_grad = (bz_grad, br_grad, bp_grad)

	"""
		The Graph

	"""

	# placeholders for the input images
	Inputs = list()
	for k in range(batchNum):
		imageBatch = list()
		for n in range(batchSize):
			imageBatch.append(tf.placeholder(tf.float64, s0))
		Inputs.append(imageBatch)

	# placeholders for the image labels
	Labels = list()
	for k in range(batchNum):
		yBatch = list()
		for n in range(batchSize):
			yBatch.append(tf.placeholder(tf.float64, (10,)))
		Labels.append(yBatch)

	# loading the images and labels
	externals = load_externals(batchNum, batchSize, Inputs, Labels)

	# getting the network saver
	saver = getSaver(K, W, U, b)
	
	# creating the graph
	batchSizeMult = tf.constant(1/batchSize, dtype = tf.float64)
	for k in range(batchNum):
		runData = list()
		"""
		runData at any index contains the following data structure:

		runData[n] = (imageConvRunData, GRUStacks)
			imageConvRunData = (A, Z)
				A = (A0, A1, A2, A3, A4, A5)
				Z = (Z0, Z2, Z4)
			GRUStacks = (Z_Stack, ZT_Stack, R_Stack, RT_Stack, P_Stack, PT_Stack, H_Stack)

		"""
		# running the network
		for n in range(batchSize):
			runData.append(network(Inputs[k][n], K, W, U, b, h, stacks, M, N, subsetNum))

		# getting the GRU gradients
		x_grad = list()
		W_grad = [tf.zeros(shape = (N, M), dtype = tf.float64)] * 3
		U_grad = [tf.zeros(shape = (N, N), dtype = tf.float64)] * 3
		b_grad = [tf.zeros(shape = (N, 1), dtype = tf.float64)] * 3
		for n in range(batchSize-1, -1, -1):
			(xn_grad, W_grad, U_grad, b_grad) = backpropGRU(runData[n], s, ks, K, W, U, b, Labels[k][n], M, N, subsetNum, W_grad, U_grad, b_grad)
			x_grad.append(tf.multiply(batchSizeMult, xn_grad))

		x_grad = x_grad[::-1]

		(Wz_grad, Wr_grad, Wp_grad) = W_grad
		(Uz_grad, Ur_grad, Up_grad) = U_grad
		(bz_grad, br_grad, bp_grad) = b_grad
		(Wz_grad, Wr_grad, Wp_grad) = (tf.multiply(batchSizeMult, Wz_grad), tf.multiply(batchSizeMult, Wr_grad), tf.multiply(batchSizeMult, Wp_grad))
		(Uz_grad, Ur_grad, Up_grad) = (tf.multiply(batchSizeMult, Uz_grad), tf.multiply(batchSizeMult, Ur_grad), tf.multiply(batchSizeMult, Up_grad))
		(bz_grad, br_grad, bp_grad) = (tf.multiply(batchSizeMult, bz_grad), tf.multiply(batchSizeMult, br_grad), tf.multiply(batchSizeMult, bp_grad))

		grad_K = list()
		for n in range(batchSize):
			# formatting inputs for CNN gradient
			A = [runData[n][0][0][j] for j in range(6)]
			Z = [runData[n][0][1][j] for j in range(3)]

			# getting the CNN gradients
			grad_K.append(backPropCNN(subsetNum, s, ks, A, Z, K, x_grad[n]))

		# summing gradients over the batch
		gradK0 = tf.add_n([grad_K[n][0] for n in range(batchSize)])
		gradK2 = tf.add_n([grad_K[n][1] for n in range(batchSize)])
		gradK4 = tf.add_n([grad_K[n][2] for n in range(batchSize)])

		# applying the gradients

		# CNN variables
		K0_op = tf.assign_sub(K0, tf.multiply(learningRate, gradK0))
		K2_op = tf.assign_sub(K2, tf.multiply(learningRate, gradK2))
		K4_op = tf.assign_sub(K4, tf.multiply(learningRate, gradK4))

		# GRU variables
		Wz_op = tf.assign_sub(Wz, tf.multiply(learningRate, Wz_grad))
		Wr_op = tf.assign_sub(Wr, tf.multiply(learningRate, Wr_grad))
		Wp_op = tf.assign_sub(Wp, tf.multiply(learningRate, Wp_grad))

		Uz_op = tf.assign_sub(Uz, tf.multiply(learningRate, Uz_grad))
		Ur_op = tf.assign_sub(Ur, tf.multiply(learningRate, Ur_grad))
		Up_op = tf.assign_sub(Up, tf.multiply(learningRate, Up_grad))

		bz_op = tf.assign_sub(bz, tf.multiply(learningRate, bz_grad))
		br_op = tf.assign_sub(br, tf.multiply(learningRate, br_grad))
		bp_op = tf.assign_sub(bp, tf.multiply(learningRate, bp_grad))

		run_op = tf.group(
			K0_op, K2_op, K4_op,
			Wz_op, Wr_op, Wp_op,
			Uz_op, Ur_op, Up_op,
			bz_op, br_op, bp_op
		)

		runTrainSession(run_op, saver, externals, savePath, loadPath)
		loadPath = savePath

	return

def runNetwork(loadPath, imageLabel, imageNum):
	# resetting the current graph
	tf.reset_default_graph()

	# hyper-parameters

	subsetNum = 4	# Number of subsets per image
	subsetSize = (14, 14, 1)	# Size of each image subset

	M = 50	# dimension of feature map
	N = 10	# number of labels

	"""
		Creating the CNN variables

	"""

	s0 = (1, 14, 14, 1)
	s1 = (1, 12, 12, 10)
	s2 = (1, 6, 6, 10)
	s3 = (1, 4, 4, 25)
	s4 = (1, 2, 2, 25)
	s5 = (1, 1, 1, 50)
	s = (s0, s1, s2, s3, s4, s5)

	ks0 = (3, 3, 1, 10)
	ks2 = (3, 3, 10, 25)
	ks4 = (2, 2, 25, 50)
	ks = (ks0, ks2, ks4)

	K0 = tf.get_variable(name = "K0", shape = ks0, initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	K2 = tf.get_variable(name = "K2", shape = ks2, initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	K4 = tf.get_variable(name = "K4", shape = ks4, initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)

	"""
		Creating the GRU cell variables

	"""

	# W matricies are multiplied with h(t-1)
	Wz = tf.get_variable(name = "Wz", shape = (N, M), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	Wr = tf.get_variable(name = "Wr", shape = (N, M), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	Wp = tf.get_variable(name = "Wp", shape = (N, M), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)

	# U matricies are multiplied with x(t-1)
	Uz = tf.get_variable(name = "Uz", shape = (N, N), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	Ur = tf.get_variable(name = "Ur", shape = (N, N), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	Up = tf.get_variable(name = "Up", shape = (N, N), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	
	# b vectors are N dimensional biases
	bz = tf.get_variable(name = "bz", shape = (N, 1), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	br = tf.get_variable(name = "br", shape = (N, 1), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	bp = tf.get_variable(name = "bp", shape = (N, 1), initializer = tf.truncated_normal_initializer(stddev = 0.1), dtype = tf.float64)
	
	# packing the network parameters
	K = (K0, K2, K4)
	W = (Wz, Wr, Wp)
	U = (Uz, Ur, Up)
	b = (bz, br, bp)

	# state h starts as a constant of 0s
	h = tf.zeros(name = "h", shape = (N, 1), dtype = tf.float64)

	# initializing the GRU internal state stacks
	Z_Stack = list()
	ZT_Stack = list()
	R_Stack = list()
	RT_Stack = list()
	P_Stack = list()
	PT_Stack = list()
	H_Stack = list()
	stacks = (Z_Stack, ZT_Stack, R_Stack, RT_Stack, P_Stack, PT_Stack, H_Stack)

	"""
		The Graph

	"""

	# placeholders for the input image
	Input = tf.placeholder(tf.float64, (subsetNum,) + subsetSize)

	# placeholders for the image label
	Label = tf.placeholder(tf.float64, (10,))

	# creating the graph
	h = network(Input, K, W, U, b, h, stacks, M, N, subsetNum)[1][6][subsetNum-1]
	h = tf.multiply(tf.constant(0.5, tf.float64), tf.add(h, tf.constant(1, dtype = tf.float64)))

	externals = dict()
	with open(dataPath + str(imageLabel) + '/Image_' + "{:04d}".format(imageNum), "rb") as f:
		externals[Input] = np.transpose(pickle.load(f), (0, 2, 3, 1))
	externals[Label] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	externals[Label][imageLabel] = 1

	saver = getSaver(K, W, U, b)

	with tf.Session() as sess:
		saver.restore(sess, loadPath)

		writer = tf.summary.FileWriter('graphs', sess.graph)

		print(sess.run(h, feed_dict = externals))

	return

def main():

	train = True
	test = True

	if train:

		learningRate = 0.0002

		batchNum = 100	# Number of batchs trained on
		batchSize = 1	# Number of images per batch

		savePath = "./saves/testing.ckpt"
		loadPath = None#"./saves/testing.ckpt"

		trainNetwork(learningRate, batchNum, batchSize, savePath, loadPath)

	if test:

		loadPath = "./saves/testing.ckpt"
		imageLabel = 3
		imageNum = 0

		runNetwork(loadPath, imageLabel, imageNum)

	return

if __name__ == "__main__":
	main()

	"""
		Command to view tensorboard:
		cd C:\\Users\\dmorl\\Desktop\\File_Folder\\coding\\Python\\Computer Vision\\Tensorflow\\TF-ICIS\\graphs\\
		tensorboard --logdir=.\\
		
	"""