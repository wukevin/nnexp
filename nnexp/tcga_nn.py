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
# np.set_printoptions(threshold=np.inf)

import tcga_imaging
import tcga_parser

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
        self.per_obs_shape = 328934 * 2 # Thsi represents the size of the flattened 2d array
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

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def softmax(patients):
    """
    The dimension of each patient's input vector is (1, 328934, 3). Let's actually simplify this
    to be just (328934, 3) instead
    This follows the pattern set forth at: https://www.tensorflow.org/get_started/mnist/pros
    """
    sess = tf.InteractiveSession()
    # Make sure the tensors we use for data and for one-hot truth are of consistent/correct length
    expression_data = ExpressionDataOneDimensional(patients)
    shape = 328934 * 2

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

def main():
    """
    """
    # Locate and load in the pickled patient records and the images
    patient_files = glob.glob(os.path.join(tcga_parser.DATA_ROOT, "tcga_patient_objects", "TCGA*.pickled"))
    assert len(patient_files) > 0
    patients = []
    for pickled_object in patient_files:
        with open(pickled_object, 'rb') as handle:
            patients.append(pickle.load(handle))
    images = glob.glob(os.path.join(tcga_imaging.IMAGES_DIR, "*.png"))
    barcodes_with_images = sortedcontainers.SortedSet([re.findall(tcga_parser.TCGA_BARCODE_REGEX, os.path.basename(x))[0] for x in images])
    # Filter the list of patients by those which we have images for and sort them
    # by alphebetical ordering of their barcodes
    patients = sortedcontainers.SortedList([x for x in patients if x.barcode in barcodes_with_images], key=lambda x: x.barcode)
    assert len(patients) > 0

    # Build the one-hot encoding table, and determine whcih of the patients this table represents
    onehot_tensor, patients_filtered = build_one_hot_encoding(patients)

    softmax(patients_filtered)

if __name__ == "__main__":
    main()
