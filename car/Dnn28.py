import numpy as np
import math
import tensorflow as tf

class Dnn28:
	def __init__(self, nclass, train_id):
		self.nclass = nclass
		self.train_id = train_id

		# Input 모델을 생성한다.
		self.input = tf.placeholder(tf.float32, [None, 784])
		# loss와 optimizer를 정의한다.
		self.output = tf.placeholder(tf.float32, [None, self.nclass])
		# Deep Neural Networks 그래프를 생성한다.
		self.output_conv, self.keep_prob = self.deepnn(self.input)
		# Cross Entropy를 비용함수(loss function)으로 정의하고, AdamOptimizer를 이용해서 비용 함수를 최소화한다.
		self.cross_entropy = tf.reduce_mean(
					tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output, logits=self.output_conv))
					#tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=self.output_conv))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
		# 정확도를 평가
		self.output_conv_max = tf.argmax(self.output_conv, 1)
		#self.normal_output = tf.contrib.layers.batch_norm(self.output_conv)

		self.output_max       = tf.argmax(self.output, 1)
		self.correct_prediction = tf.equal(self.output_conv_max, self.output_max)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		self.saver = tf.train.Saver()
		self.sess  = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def  deepnn(self, input):
		# RGB 이미지라면 3차원, RGBA라면 4차원 이미지 일 것이다.
		input_image = tf.reshape(input, [-1, 28, 28, 1])

		# 첫번째 convolutional layer - 하나의 grayscale 이미지를 32개의 특징들(feature)으로 맵핑(maping)한다.
		W_conv1 = self.weight_variable([5, 5, 1, 32])
		b_conv1 = self.bias_variable([32])
		h_conv1 = tf.nn.relu(self.conv2d(input_image, W_conv1) + b_conv1)

		# Pooling layer - 2X만큼 downsample한다.
		h_pool1 = self.max_pool_2x2(h_conv1)

		# 두번째 convolutional layer -- 32개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)한다.
		W_conv2 = self.weight_variable([5, 5, 32, 64])
		b_conv2 = self.bias_variable([64])
		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

		# 두번째 pooling layer.
		h_pool2 = self.max_pool_2x2(h_conv2)

		# Fully Connected Layer 1 -- 2번의 downsampling 이후에, 우리의 28x28 이미지는 7x7x64 특징들(feature map)이 된다.
		# 이를 1024개의 특징들로 맵핑(maping)한다.
		W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
		b_fc1 = self.bias_variable([1024])

		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		# Dropout - 모델의 복잡도를 컨트롤한다. 특징들의 co-adaptation을 방지한다.
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		# 1024개의 특징들(feature)을 10개의 클래스-숫자 0-9-로 맵핑(maping)한다.
		W_fc2 = self.weight_variable([1024, self.nclass])
		b_fc2 = self.bias_variable([self.nclass])

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		return y_conv, keep_prob

	def restore(self):
		self.saver.restore(self.sess, "car/dnn/" +self.train_id)

	def save(self):
		self.saver.save(self.sess, "car/dnn/"+ self.train_id)

	def close(self):
		self.sess.close()

	def train(self, images, labels):
		return self.sess.run([self.train_step], feed_dict={self.input:images, self.output:labels, self.keep_prob: 0.5})

	def evalAccuracy(self, images, labels):
		return self.sess.run([self.accuracy], feed_dict={self.input:images, self.output:labels, self.keep_prob: 1})

	def evalOutput(self, image):
		label = np.zeros(self.nclass, np.uint8) #  dumnmy
		r1, r2 = self.sess.run([self.output_conv_max, self.output_conv],
				feed_dict={self.input:[image], self.output:[label], self.keep_prob: 1})
		#print('r1={},r2={}'.format(r1, r2))
		alist = [(i, r2[0][i]) for i in range(len(r2[0])) ]
		alist = sorted(alist, key=lambda c: c[1], reverse=True)

		#print(alist[:5])
		return r1

	def conv2d(self, x, W):
		"""conv2d는 full stride를 가진 2d convolution layer를 반환(return)한다."""
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
		"""max_pool_2x2는 특징들(feature map)을 2X만큼 downsample한다."""
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

	def weight_variable(self, shape):
		"""weight_variable는 주어진 shape에 대한 weight variable을 생성한다."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		"""bias_variable 주어진 shape에 대한 bias variable을 생성한다."""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)
