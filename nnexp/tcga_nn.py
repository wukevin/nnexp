"""
The neural network itself!
"""
import tensorflow as tf
import sys
import os
import re
import glob
import pickle
import sortedcontainers
import numpy as np
import tqdm
import random
import time

import argparse

import tcga_imaging
import tcga_parser
import tcga_processor
import tcga_analysis

KEYS_OF_INTEREST = [
    "er_status_by_ihc",
    "pr_status_by_ihc"
    "her2_status_by_ihc"
]

class ExpressionDataOneDimensional(object):
    """
    Class that returns the expresion data as a one-dimensional vector, in chunks
    """
    def __init__(self, patients, save_for_testing=50, clinical_key="her2_status_by_ihc"):
        # Make sure the input is of the correct type
        assert all([isinstance(x, tcga_parser.TcgaPatient) for x in patients])
        # Filter out the list of inputted patients for any that do not have the given
        # clinical key
        self.key = clinical_key
        self.accepted_key_values = ["Negative", "Positive"]
        self.all_patients = [x for x in patients if self.key in x.clinical and x.clinical[self.key] in self.accepted_key_values]
        # Partition data into training and testing datasets
        self.training_patients = patients[:-save_for_testing]
        self.testing_patients = patients[-save_for_testing:]
        # Load in the actual vectors
        self.per_obs_shape = 15368 * 2 # Thsi represents the size of the flattened 2d array
        # Locate and load in the patient vectors
        self.training_expression_vectors = {}
        for patient in self.training_patients:
            np_array_location = os.path.join(
                tcga_imaging.TENSORS_DIR,
                "%s.expression.array" % patient.barcode
            )
            assert os.path.isfile(np_array_location)
            with open(np_array_location, 'rb') as handle:
                expression_array = pickle.load(handle)
            # print(expression_array.shape)
            # Remove the first 1-dimension, maintianing a 2-dimensional array
            # expression_array = np.reshape(expression_array, expression_array.shape[1:])
            # Flatten the array to be one-dimensional
            expression_array = np.reshape(expression_array, np.prod(expression_array.shape))
            assert expression_array.shape[0] == self.per_obs_shape
            self.training_expression_vectors[patient.barcode] = expression_array

        self.testing_expression_vectors = {}
        for patient in self.testing_patients:
            np_array_location = os.path.join(
                tcga_imaging.TENSORS_DIR, "%s.expression.array" % patient.barcode
            )
            with open(np_array_location, 'rb') as handle:
                expression_array = pickle.load(handle)
            expression_array = np.reshape(expression_array, np.prod(expression_array.shape))
            assert expression_array.shape[0] == self.per_obs_shape
            self.testing_expression_vectors[patient.barcode] = expression_array

        self.index = 0

    def next_training_batch(self, n):
        # next_index = min(self.index + n, len(self.training_patients))
        # subsetted_patients = [x for x in self.training_patients[self.index:next_index]]
        subsetted_patients = random.sample(self.training_patients, n)
        assert len(subsetted_patients) > 0
        subsetted_data = np.vstack([self.training_expression_vectors[x.barcode] for x in subsetted_patients])
        assert subsetted_data.shape[1] == self.per_obs_shape
        # Buidl the one-hot truth tensor
        onehot = np.zeros((subsetted_data.shape[0], 2), dtype=np.float32)
        for index, x in enumerate(subsetted_patients):
            onehot[index][self.accepted_key_values.index(x.clinical[self.key])] = 1.0
        # Increment the index
        # self.index = (self.index + n) % len(self.training_patients)

        return subsetted_data, onehot

    def testing_batch(self):
        testing_data = np.vstack([self.testing_expression_vectors[x.barcode] for x in self.testing_patients])
        onehot = np.zeros((testing_data.shape[0], 2), dtype=np.float32)
        for index, x in enumerate(self.testing_patients):
            onehot[index][self.accepted_key_values.index(x.clinical[self.key])] = 1.0

        return testing_data, onehot


class ExpressionDataThreeDimensional(object):
    """
    Derivative of the above function that, instead of dealing with one dimensional vectors,
    deals in three dimensions
    """
    def __init__(self, patients, save_for_testing=50, start_of_testing=0, clinical_key="her2_status_by_ihc"):
        self.key = clinical_key
        self.accepted_key_values = ["Negative", "Positive"]
        self.all_patients = [x for x in patients if self.key in x.clinical and x.clinical[self.key] in self.accepted_key_values]
        # Partition data into training and testing datasets
        self.testing_patients = patients[start_of_testing:save_for_testing+start_of_testing]
        self.training_patients = [x for x in patients if x not in self.testing_patients]
        # Load in the actual vectors
        self.per_obs_shape = (11, 1600, 2)  # The size of the 3D array
        # Locate and load in the patient vectors
        self.training_expression_vectors = {}
        for patient in self.training_patients:
            np_array_location = os.path.join(
                tcga_imaging.TENSORS_DIR, "{0}.expression.array".format(patient.barcode)
            )
            assert os.path.isfile(np_array_location)
            with open(np_array_location, 'rb') as handle:
                expression_array = pickle.load(handle)
            assert expression_array.shape == self.per_obs_shape
            self.training_expression_vectors[patient.barcode] = expression_array
        # ... and the training patient expression vectors
        self.testing_expression_vectors = {}
        for patient in self.testing_patients:
            np_array_location = os.path.join(
                tcga_imaging.TENSORS_DIR, "%s.expression.array" % patient.barcode
            )
            with open(np_array_location, 'rb') as handle:
                expression_array = pickle.load(handle)
            assert expression_array.shape == self.per_obs_shape
            self.testing_expression_vectors[patient.barcode] = expression_array

        self.index = 0

        # Read in the ranges that describes the maximum/minimum values for each datatype
        self.ranges = {}
        with open(tcga_analysis.RANGES_FILE, 'r') as handle:
            for line in handle:
                line = line.rstrip()
                datatype, minimum, _delim, maximum = line.split()
                self.ranges[datatype] = (np.float32(minimum), np.float32(maximum))

    def next_training_batch(self, n=50, distort=True, sd_prop=0.003):
        """
        Returns the next batch of data, reshaped into a stright 1-dimensional vector. We will re-format this into
        a 3 dimensional vector again later

        If distortion is enabled, then we use the given <sd_prop> to generate a normal
        distribution around 1, which we then use to scale the data to add "noise" on a
        per-entry basis. For example, if using a sd_prop of 0.01, we generated a random
        number o.99, then the value at that specific cell would be scaled to 0.99x its
        original value. This is randomly done for each cell
        """
        # next_index = min(self.index + n, len(self.training_patients))
        # subsetted_patients = [x for x in self.training_patients[self.index:next_index]]
        subsetted_patients = random.sample(self.training_patients, n)
        assert subsetted_patients
        expression_data = []
        for patient in subsetted_patients:
            # Each expression vector is 11 x 1600 x 2
            data = self.training_expression_vectors[patient.barcode]
            original_shape = data.shape
            if distort:
                # Distort the data a little bit. The 0th channel is CNV, and the 1st channel is RNA
                noise = np.random.normal(
                    loc=1.0,
                    scale=sd_prop,
                    size=data.shape
                )
                data = data * noise
                # Normalize the data to make sure min/max is still true
                # data[:, :, 0][data[:, :, 0] > cnv_max] = cnv_max
                # data[:, :, 0][data[:, :, 0] < cnv_min] = cnv_min
                # data[:, :, 1][data[:, :, 1] > rna_max] = rna_max
                # data[:, :, 1][data[:, :, 1] < rna_min] = rna_min
            assert data.shape == original_shape

            expression_data.append(data.flatten())
        subsetted_data = np.vstack(expression_data)
        # subsetted_data = np.vstack([self.training_expression_vectors[x.barcode].flatten() for x in subsetted_patients])
        # Build the one-hot truth tensor
        onehot = np.zeros((subsetted_data.shape[0], 2), dtype=np.float32)
        for index, x in enumerate(subsetted_patients):
            onehot[index][self.accepted_key_values.index(x.clinical[self.key])] = 1.0
        # Increment the index
        # self.index = (self.index + n) % len(self.training_patients)

        return subsetted_data, onehot

    def testing_batch(self):
        testing_data = np.vstack([self.testing_expression_vectors[x.barcode].flatten() for x in self.testing_patients])
        onehot = np.zeros((testing_data.shape[0], 2), dtype=np.float32)
        for index, x in enumerate(self.testing_patients):
            onehot[index][self.accepted_key_values.index(x.clinical[self.key])] = 1.0

        return testing_data, onehot



def build_one_hot_encoding(patients):
    """
    Builds a one-hot encoding table for the given list of patients
    """
    # Sanity check the input data
    for patient in patients:
        assert isinstance(patient, tcga_parser.TcgaPatient)

    # Define which keys we're interested in, and that they are present
    # across all the patients
    # for key in keys_of_interest:
    #     for patient in patients:
    #         assert key in patient.clinical

    allowed_values = ["Positive", "Negative"]
    patients_filtered = [x for x in patients if x.clinical['her2_status_by_ihc'] in allowed_values]
    # Build the matrix
    onehot = tf.one_hot([allowed_values.index(x.clinical['her2_status_by_ihc']) for x in patients_filtered], 2)
    # For visually checking the one-hot vector
    # onehot_np = np.array(onehot.eval())
    # print(type(onehot_np))
    # print(onehot_np)
    return onehot, patients_filtered


def build_patients_vector():
    pass

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, strides=[1,1,1,1], pad="SAME"):
    return tf.nn.conv2d(x, W, strides=strides, padding=pad)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  # ksize is batch size, height, width, num channels

def multilayer_cnn(patients, kth=2, ksize=40, training_iters=5000, training_size=25):
    """
    Dimension of the data is (11, 1600, 2)
    """
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 11 * 1600 * 2])
    y_ = tf.placeholder(tf.float32, [None, 2])
    x_image = tf.reshape(x, [-1, 11, 1600, 2])  # Reshape it back to the image that we want

    # First convolutional layer.
    # Start with a 1x1 convolution, https://iamaaditya.github.io/2016/03/one-by-one-convolution/
    first_num_features = 6
    W_conv1 = weight_variable([1, 1, 2, first_num_features]) # first two are patch size, then input channels, then number of output features
    b_conv1 = bias_variable([first_num_features])  # There is a bias for every output channel
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # The output of this is a 11x1600xnum_features (n channels from the original 2)
    # ksize is [batch, height, width, channels]
    # This pooling is mostly to just even out the dimensions to be a pretty 10 x 1600 x num_features
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 1, 1], strides=[1, 1, 1, 1], padding='VALID')  # Reduces to 10 x 1600 x 8
    print("Pooled first layer:", h_pool1.get_shape())  # (?, 10, 1600, first_num_features)

    # Second convolutional layer
    second_num_features = 24
    second_patch_height, second_patch_width = 1, 32
    W_conv2 = weight_variable([second_patch_height, second_patch_width, first_num_features, second_num_features])  # outputs 32 features for each 10x10 patch
    b_conv2 = bias_variable([second_num_features])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    second_pool_window = 8 # Originally 5
    # Reduce from 10x1600xfirst_num_features to be 1600 / second_pool_window long
    h_pool2 = tf.nn.max_pool(
        h_conv2,
        ksize=[1, 1, second_pool_window, 1],
        strides=[1, 1, second_pool_window, 1],
        padding='SAME'
    )
    print("Pooled second layer:", h_pool2.get_shape()) # (?, 10, 400, second_num_features)

    # Densely connected layer
    final_shape = [x for x in h_pool2.get_shape().as_list() if x is not None]
    flattened = int(np.product(final_shape))
    print("Final flattened:", flattened)
    dense_num_features = 4096 # previously 4096
    # flattened = int(10 * 1600 / second_pool_window * second_num_features)
    W_fc1 = weight_variable([flattened, dense_num_features])
    b_fc1 = bias_variable([dense_num_features])
    h_pool2_flat = tf.reshape(h_pool2, [-1, flattened])
    h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout
    W_fc2 = weight_variable([dense_num_features, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # A learning rate of 1e-4 seems to cause us to never really converge on a good solution
    # the default value (1e-3?) (with 1100 trainign iterations) seems to work well
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))  # Number of things correct

    # Run
    logfile = open("cnn.{0}.log".format(kth), 'w')
    # testing_batch_size = 40
    # k_validations = round(len(patients) / testing_batch_size)
    # for k in range(k_validations):
        # print("{0} of {1} cross validation steps".format(k, k_validations))
    expression_data = ExpressionDataThreeDimensional(patients, save_for_testing=ksize, start_of_testing=ksize*kth)
    sess.run(tf.global_variables_initializer())
    for _i in range(training_iters):
        batch_xs, batch_ys = expression_data.next_training_batch(training_size)
        if _i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            status_update = "step %d, training accuracy %g" % (_i, train_accuracy)
            logfile.write(status_update + "\n")
            print(status_update)
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    test_data, test_truth = expression_data.testing_batch()
    num_right = 0
    for test_data_single, test_truth_single, test_patient in zip(test_data, test_truth, expression_data.testing_patients):
        acc = accuracy.eval(feed_dict={x: [test_data_single], y_: [test_truth_single], keep_prob: 1.0})
        line = "\t".join([test_patient.barcode, test_patient.clinical['her2_status_by_ihc'], str(int(acc))])
        print(line)
        logfile.write(line + "\n")
        num_right += 1 if int(acc) == 1 else 0
    # final_accuracy = accuracy.eval(feed_dict={x: test_data, y_: test_truth, keep_prob: 1.0})
    # final_num_right = num_correct.eval(feed_dict={x: test_data, y_: test_truth, keep_prob: 1.0})
    final_accuracy = num_right / len(expression_data.testing_patients)
    status_update = "final test accuracy: %g - %i / %i" % (final_accuracy, num_right, len(expression_data.testing_patients))    
    print(status_update)
    # print(correct_prediction.eval())
    # print out a breakdown
    # for truth, prediction in zip(test_truth, correct_prediction.eval()):
    #     print(truth, prediction)
    logfile.write(status_update + "\n")
    logfile.close()  # Close the filehandle

def softmax(patients):
    """
    The dimension of each patient's input vector is (1, 15368, 3). Let's actually simplify this
    to be just (15368, 3) instead
    This follows the pattern set forth at: https://www.tensorflow.org/get_started/mnist/pros
    """
    sess = tf.InteractiveSession()
    # Make sure the tensors we use for data and for one-hot truth are of consistent/correct length
    expression_data = ExpressionDataOneDimensional(patients)
    shape = 15368 * 2

    x = tf.placeholder(tf.float32, [None, shape])

    weight = tf.Variable(tf.zeros([shape, 2]))
    bias = tf.Variable(tf.zeros([2]))

    y = tf.nn.softmax(tf.matmul(x, weight) + bias)

    # These are the correct answers
    y_ = tf.placeholder(tf.float32, [None, 2])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    tf.global_variables_initializer().run()

    # Training
    for _ in tqdm.tqdm(range(2000)):
        batch_xs, batch_ys = expression_data.next_training_batch(20)
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

    # Evaluate
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_data, test_truth = expression_data.testing_batch()
    print(sess.run(accuracy, feed_dict={x: test_data, y_: test_truth}))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--kth", type=int, required=True, help="The kth block to use as a truth set. 0-indexed")
    parser.add_argument("-s", "--size", type=int, default=50, help="Size of the blocks for truth sets")
    parser.add_argument("-i", "--iter", type=int, default=2000, help="Number of training iterations to run")  # Default used to be 5000
    parser.add_argument("-n", "--itersize", type=int, default=25, help="Number of samples to run per training iteration")
    return parser


def main():
    """
    """
    parser = build_parser()
    args = parser.parse_args()

    # Locate and load in the pickled patient records and the images
    patients = tcga_processor.load_tcga_objects()
    images = glob.glob(os.path.join(tcga_imaging.TENSORS_DIR, "*.expression.array"))
    barcodes_with_images = sortedcontainers.SortedSet([re.findall(tcga_parser.TCGA_BARCODE_REGEX, os.path.basename(x))[0] for x in images])
    # Filter the list of patients by those which we have images for and sort them
    # by alphebetical ordering of their barcodes
    patients = sortedcontainers.SortedList([x for x in patients if x.barcode in barcodes_with_images], key=lambda x: x.barcode)
    assert patients  # Make sure the patients list is not empty

    # Build the one-hot encoding table, and determine whcih of the patients this table represents
    onehot_tensor, patients_filtered = build_one_hot_encoding(patients)

    # softmax(patients_filtered)
    multilayer_cnn(patients_filtered, args.kth, args.size, training_iters=args.iter, training_size=args.itersize)

if __name__ == "__main__":
    main()
