from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
	"""
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""
	num_examples = inputs.shape[0]
	in_height = inputs.shape[1]
	in_width = inputs.shape[2]
	input_in_channels = inputs.shape[3]

	filter_height = filters.shape[0]
	filter_width = filters.shape[1]
	filter_in_channels = filters.shape[2]
	filter_out_channels = filters.shape[3]

	num_examples_stride = 1
	strideY = 1
	strideX = 1
	channels_stride = 1

	# filters h, w, channel, number of filters
	if filters.shape[2] != inputs.shape[3] :
		raise ValueError("Filter and inputs have different dimensions!! ")

	# Cleaning padding input
	padX = 0
	padY = 0
	if padding == 'SAME':
		padX = math.floor((filter_width - 1)/2)
		padY = math.floor((filter_height - 1)/2)
		p = ((0,0),(padX,padX), (padX,padY), (0,0))
		inputs = np.pad(inputs, p)

	# Calculate output dimensions
	out_h = int((in_height + 2*padY - filter_height) / strideY + 1)
	out_w = int((in_width + 2*padX - filter_width) / strideX + 1)

	if padding == 'SAME':
		filtered_image = np.zeros((num_examples, in_height, in_width, filter_out_channels))
		for f in range(filter_out_channels):
			for im_n in range(num_examples):
				for i in range(padY, padY + out_h):
					for j in range(padX, padX + out_w):
						for color_channel in range(filter_in_channels):
							image_extract = inputs[im_n, i-padY: i - padY + filter_height, j-padX: j - padX + filter_width, color_channel]
							res = np.multiply(image_extract, filters[:,:,color_channel, f])
							filtered_image[im_n][i-padY][j-padX][f] += np.sum(res)

	if padding == 'VALID':
		filtered_image = np.zeros((num_examples, out_h, out_w, filter_out_channels))
		for f in range(filter_out_channels):
			for im_n in range(num_examples):
				for i in range(padY, padY + out_h):
					for j in range(padX, padX + out_w):
						for color_channel in range(filter_in_channels):
							image_extract = inputs[im_n, i - padY: i + filter_height, j - padX: j + filter_width, color_channel]
							res = np.multiply(image_extract, filters[:,:,color_channel, f])
							filtered_image[im_n][i-padY][j-padX][f] += np.sum(res)
	# PLEASE RETURN A TENSOR. HINT: tf.convert_to_tensor(your_array, dtype = tf.float32)
	print('buff')
	return(tf.convert_to_tensor(filtered_image, dtype = tf.float32))
	pass


def same_test_0():
	'''
	Simple test using SAME padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
	print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())

def valid_test_0():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	#changed
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	#changed
	filters = tf.Variable(tf.random.truncated_normal([1, 1, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def main():
	# TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output
	same_test_0()
	valid_test_0()
	valid_test_1()
	valid_test_2()
	return

if __name__ == '__main__':
	main()
