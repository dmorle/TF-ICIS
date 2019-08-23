import numpy as np
import tensorflow as tf

def convBackprop_Slicing(s0, s1, ks0):

	A0 = np.empty(s0)
	A1 = np.arange(s1[0]*s1[1]*s1[2]).reshape(s1)
	K0 = np.arange(ks0[0]*ks0[1]*ks0[2]*ks0[3]).reshape(ks0)

	for l in range(s0[0]):
		for m in range(s0[1]):
			for n in range(s0[2]):
				Al0 = max(0, l-ks0[0]+1)
				Al1 = min(s1[0], l+1)
				Am0 = max(0, m-ks0[1]+1)
				Am1 = min(s1[1], m+1)

				Kl0 = max(0, l-s1[0]+1)
				Kl1 = min(ks0[0], l+1)
				Km0 = max(0, m-s1[1]+1)
				Km1 = min(ks0[1], m+1)

				A0[l, m, n] = np.tensordot(
					A1[Al0:Al1, Am0:Am1], 
					K0[Kl0:Kl1, Km0:Km1, n][::-1, ::-1], 
					3
				)

	return A0

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

			gradZ0 = tf.pad(gradZ0, tf.constant([[0, 0], [ks0[0]-1, ks0[0]-1], [ks0[1]-1, ks0[1]-1], [0, 0]]), "CONSTANT")
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

def runBackprop_A():
	# s0 = (6, 6, 3)		# height, width, depth
	# s1 = (4, 4, 4)		# height, width, depth
	# ks0 = (3, 3, 3, 4)	# height, width, in_depth, out_depth

	# print(convBackprop_Slicing(s0, s1, ks0)[::, ::, 0])

	s0 = [1, 6, 6, 3]	# num of batches, height, width, channels
	s1 = [1, 4, 4, 4]	# num of batches, height, width, channels
	ks0 = [3, 3, 3, 4]	# height, width, in_depth, out_depth

	gradZ0 = tf.reshape(tf.cast(tf.range(s1[0]*s1[1]*s1[2]*s1[3]), tf.float64), s1)
	K0 = tf.reshape(tf.cast(tf.range(ks0[0]*ks0[1]*ks0[2]*ks0[3]), tf.float64), ks0)

	gradA0 = convLayerBackprop(s0, s1, ks0, K0, gradZ0)
	gradA0_TF = tf.nn.conv2d_backprop_input(s0, K0, gradZ0, [1, 1, 1, 1], "VALID", False)
	with tf.Session() as sess:
		print(sess.run(gradA0)[0, ::, ::, 0])
		print(sess.run(gradA0_TF)[0, ::, ::, 0])
		pass
	return

def main():
	runBackprop_A()
	return

if __name__ == "__main__":
	main()