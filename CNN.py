import tensorflow as tf
import numpy as np

# tensorflow version of numpy's repeat
def tensorRepeat(X, n):
	if type(n) is int:
		s = X.shape

		X = tf.tile(X, [1] + [n for i in range(len(s)-1)])
		X = tf.reshape(X, [-1, 1])
		X = tf.tile(X, [1, n])

		X = tf.reshape(X, [i*n for i in s])

		return X

	else:
		s = X.shape

		X = tf.tile(X, [1] + [n[i] for i in range(len(s)-1)])
		X = tf.reshape(X, [-1, 1])
		X = tf.tile(X, [1, n[-1]])

		X = tf.reshape(X, [s[i]*n[i] for i in range(len(s))])
		return X

# A0 is the input tensor to the pooling layer in forward prop
# A1 is the output tensor of the pooling layer (A1 = pool(A0))
# gradA1 is the tensor representing the gradient wrt each of the elements in A1
# (m, n) is the stride and size of the pooling
# returns gradA0
def poolLayerBackprop(A0, A1, gradA1, m, n, name = None):
	if name:
		with tf.name_scope(name):
			A1 = tensorRepeat(A1, [1, m, n, 1])
			gradA1 = tensorRepeat(gradA1, [1, m, n, 1])
			gradA0 = tf.multiply(gradA1, tf.cast(tf.equal(A0, A1), tf.float64))
			return gradA0
	else:
		with tf.name_scope("Pool_Layer_Backprop"):
			A1 = tensorRepeat(A1, [1, m, n, 1])
			gradA1 = tensorRepeat(gradA1, [1, m, n, 1])
			gradA0 = tf.multiply(gradA1, tf.cast(tf.equal(A0, A1), tf.float64))
			return gradA0

# s0 is the size of the input tensor for forward prop
# s1 is the size of the output tensor for forward prop
# ks0 is the size of kernel used for A1 = RELU(conv(A0, K0))
# K0 is the kernel used in the forward prop conv
# gradZ0 is the gradient tensor representing conv(A0, K0) *Note, before RELU
# returns (gradA0, gradK0)
def convLayerBackprop(s0, s1, ks0, K0, gradZ0, name = None):
	if not name:
		with tf.name_scope("ConvBackprop"):
			# A1 = tf.reshape(tf.range(s1[0]*s1[1]*s1[2]*s1[3], dtype = tf.float64), s1)
			# K0 = tf.reshape(tf.range(ks0[0]*ks0[1]*ks0[2]*ks0[3], dtype = tf.float64), ks0)

			gradZ0 = tf.pad(gradZ0, tf.constant([[0, 0], [ks0[0]-1, ks0[0]-1], [ks0[1]-1, ks0[1]-1], [0, 0]]))
			K0 = tf.transpose(K0[::-1, ::-1], (0, 1, 3, 2))

			return tf.nn.conv2d(gradZ0, K0, [1, 1, 1, 1], "VALID")
	else:
		with tf.name_scope(name):
			gradZ0 = tf.multiply(tf.cast(tf.greater(Z0, 0), tf.float64), gradA1)

			# A1 = tf.reshape(tf.range(s1[0]*s1[1]*s1[2]*s1[3], dtype = tf.float64), s1)
			# K0 = tf.reshape(tf.range(ks0[0]*ks0[1]*ks0[2]*ks0[3], dtype = tf.float64), ks0)

			gradZ0 = tf.pad(gradZ0, tf.constant([[0, 0], [ks0[0]-1, ks0[0]-1], [ks0[1]-1, ks0[1]-1], [0, 0]]))
			K0 = tf.transpose(K0[::-1, ::-1], (0, 1, 3, 2))

			return tf.nn.conv2d(gradZ0, K0, [1, 1, 1, 1], "VALID")

# returns (
#	gradK0,
#	gradK2,
#	gradK4
# )
def backPropCNN(subsetNum, s, ks, A, Z, K, gradA5):

	# unpacking inputs
	(A0, A1, A2, A3, A4, A5) = A
	(Z0, Z2, Z4) = Z
	(K0, K2, K4) = K

	(s0, s1, s2, s3, s4, s5) = s
	(ks0, ks2, ks4) = ks

	gradK0 = None
	gradK2 = None
	gradK4 = None

	# starting backpropagation

	gradZ4 = tf.multiply(tf.cast(tf.greater(Z4, 0), tf.float64), gradA5)
	gradK4 = tf.nn.conv2d_backprop_filter(A4, ks4, gradZ4, [1, 1, 1, 1], "VALID", False)
	gradA4 = tf.nn.conv2d_backprop_input(s4, K4, gradZ4, [1, 1, 1, 1], "VALID", False)
	# gradA4 = convLayerBackprop(s4, s5, ks4, K4, gradZ4) - my implimentation

	gradA3 = poolLayerBackprop(A3, A4, gradA4, 2, 2, name = "A3_Gradient_Pool")

	gradZ2 = tf.multiply(tf.cast(tf.greater(Z2, 0), tf.float64), gradA3)
	gradK2 = tf.nn.conv2d_backprop_filter(A2, ks2, gradZ2, [1, 1, 1, 1], "VALID", False)
	gradA2 = tf.nn.conv2d_backprop_input(s2, K2, gradZ2, [1, 1, 1, 1], "VALID", False)
	# gradA2 = convLayerBackprop(s2, s3, ks2, K2, gradZ2) - my implimentation

	gradA1 = poolLayerBackprop(A1, A2, gradA2, 2, 2, name = "A1_Gradient_Pool")

	gradZ0 = tf.multiply(tf.cast(tf.greater(Z0, 0), tf.float64), gradA1)
	gradK0 = tf.nn.conv2d_backprop_filter(A0, ks0, gradZ0, [1, 1, 1, 1], "VALID", False)

	# finished all the activation backpropagation
	return (gradK0, gradK2, gradK4)

def CNN(A0, K):

	# unpacking inputs
	(K0, K2, K4) = K

	with tf.name_scope("Conv1"):
		Z0 = tf.nn.conv2d(A0, K0, (1, 1, 1, 1), "VALID")
		A1 = tf.nn.relu(Z0)

	with tf.name_scope("Pool2"):
		A2 = tf.nn.max_pool(A1, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

	with tf.name_scope("Conv3"):
		Z2 = tf.nn.conv2d(A2, K2, (1, 1, 1, 1), "VALID")
		A3 = tf.nn.relu(Z2)

	with tf.name_scope("Pool4"):
		A4 = tf.nn.max_pool(A3, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

	with tf.name_scope("Conv5"):
		Z4 = tf.nn.conv2d(A4, K4, (1, 1, 1, 1), "VALID")
		A5 = tf.nn.relu(Z4)

	return (
		(A0, A1, A2, A3, A4, A5),
		(Z0, Z2, Z4)
	)