# -*- coding: utf-8 -*-

import numpy as np


def normalize(features):
    """Normalize features."""
    # Copy original array to prevent it from changes.
    features_normalized = np.copy(features).astype(float)
    # Get average values for each feature (column) in X.
    features_mean = np.mean(features, 0)
    # Calculate the standard deviation for each feature.
    features_deviation = np.std(features, 0)
    # Subtract mean values from each feature (column) of every example(row)
    # to make all features be spread around zero
    if features.shape[0] > 1:
        features_normalized - features_mean

    # Normalize each feature values so that all features are close to [-1:1] boundaries.
    # Also prevent division by zero error.
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation


def generate_sinusoids(dataset, sinusoid_degree):
    """Extends dataset with sinusoid features."""
    # Create sinusoids matrix.
    num_examples = dataset.shape[0]
    sinusoids = np.empty((num_examples, 0))

    # Generate sinusoid features of specified degree.
    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    return sinusoids


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """Extends dataset with polynomial features of certain degree.
    
    Returns a new feature array with more features, comprising of
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.
    """
    # Split features on two halves.
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    # Extract se
    num_examples_1, num_features_1 = dataset_1.shape
    num_examples_2, num_features_2 = dataset_2.shape

    # Check if two sets have equal amount of rows.
    if num_examples_1 != num_examples_2:
        raise ValueError('can NOT generate polynomials for two sets with different number of rows!')
    # Check if at least one set has features.
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('can NOT generate polynomials for two sets with no columns!')

    # Replace empty set with non-empty one.
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    # Make sure that sets have the same number of features in order to be able to multiply then.
    num_features = num_features_1 if num_features_1 < num_features_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    # Create polynomials matrix.
    polynomials = np.empty((num_examples_1, 0))

    # Generate polynomial features of specified degree.
    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    # Normalize polynomials if needed.
    if normalize_data:
        polynomials = normalize(polynomials)[0]

    return polynomials


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """Prepare dataset for training on prediction."""

    # Calculate the number of examples.
    num_examples = data.shape[0]

    # Prevent original data from being modified.
    data_processed = np.copy(data)

    # Normalize data set.
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed

    if normalize_data:
        data_normalized, features_mean, features_deviation = normalize(data_processed)

        # Replace processed data with normalized processed data.
        # We need to have normalized data below while we will adding polynormials and sinusoids.
        data_processed = data_normalized

    # Add sinusoidal features to the dataset.
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # Add polynomial features to dataset.
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # Add a column of ones to X:
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation
