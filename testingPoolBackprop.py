import tensorflow as tf
import numpy as np

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

def getGradient(A0, A1, n):
	A1 = tensorRepeat(A1, [1, n, n, 1])
	grad = tf.multiply(A0, tf.cast(tf.equal(A0, A1), tf.int32))
	return grad

def main():
	(w, x, y, z) = (1, 4, 4, 3)
	A0 = tf.constant(np.random.randint(0, 10, w*x*y*z).reshape(w, x, y, z))

	A1 = tf.nn.max_pool(A0, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
	grad = getGradient(A0, A1, 2)

	A0 = tf.transpose(A0, [0, 3, 1, 2])
	A1 = tf.transpose(A1, [0, 3, 1, 2])
	grad = tf.transpose(grad, [0, 3, 1, 2])

	with tf.Session() as sess:
		print(sess.run(A0))
		print(sess.run(A1))
		print(sess.run(grad))
	return

if __name__ == "__main__":
	main()