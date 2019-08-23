import numpy as np
import tensorflow as tf

def matHadamard(T, v):
	shape = T.shape
	m = shape[0]
	n = shape[1]
	T = tf.tensordot(v.reshape(m), T, 0)
	print(T)
	U_Tensor = np.concatenate([np.identity(m).reshape(m, m, 1) for i in range(n)], 2)
	T = tf.multiply(T, U_Tensor)
	T = tf.reduce_sum(T, 0)
	return T

def matHadamard(T, v):
	shape = T.shape
	m = shape[0]
	n = shape[1]
	T = np.tensordot(v.reshape(m), T, 0)
	print(T)
	U_Tensor = np.concatenate([np.identity(m).reshape(m, m, 1) for i in range(n)], 2)
	T = np.multiply(T, U_Tensor)
	T = np.reduce_sum(T, 0)
	return T

def main():
	# x = np.arange(12).reshape((3, 4))*2 + 1
	# print(x)
	# print("X^\n")

	# y = (np.arange(3) + 1).reshape(3, 1)
	# print(y)
	# print("Y^\n")

	# z = matHadamard(x, y)
	# print(z)
	# with tf.Session() as sess:
	#	print(sess.run(z))
	print([i+3*j for j in range(3) for i in range(3)])
	return

if __name__ == "__main__":
	main()