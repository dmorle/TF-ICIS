import pickle
import numpy as np
import tensorflow as tf

def TF_op():
	x = tf.zeros(shape = (0, 2), dtype = tf.float64)
	x = tf.concat((tf.zeros(shape = (1, 2), dtype = tf.float64), x), 0)
	return x 

def func(P, image):
	(x, y, z) = P
	print(x, y, z)
	return

def loadSubset():
	with open("subsetData/0/Image_0000", "rb") as f:
		data = pickle.load(f)

	print(data.shape)
	return

def main():
	# x = TF_op()
	# with tf.Session() as sess:
	# 	print(sess.run(x))

	#loadSubset()

	# arr = np.arange(9).reshape(3, 3)
	# print(arr)
	# for i in arr:
	# 	print(i)

	x = np.arange(8).reshape(2, 2, 2)
	print(x[0:1])
	print(x[1:2])
	return

if __name__ == "__main__":
	main()