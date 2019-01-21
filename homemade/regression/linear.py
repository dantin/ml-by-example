# -*- coding: utf-8 -*-
""" Linear Regression."""


import numpy as np
from ..utils.features import prepare_for_training


class LinearRegression():
    """Linear Regression Class."""

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        data_processed, features_mean, features_deviation = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # Initialize model parameters.
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, lambda_param=0, num_iterations=500):
        """Train linear regression."""
        # Run gradient descent.
        cost_history = self.gradient_descent(alpha, lambda_param, num_iterations)

        return self.theta, cost_history

    def gradient_descent(self, alpha, lambda_param, num_iterations):
        """Gradient descent."""
        # Initialize J_history with zeros.
        cost_history = []

        for _ in range(num_iterations):
            # Perform a single gradient step on the parameter vector theta.
            self.gradient_step(alpha, lambda_param)

            # Save the cost J in every iteration.
            cost_history.append(self.cost_function(self.data, self.labels, lambda_param))

        return cost_history

    def gradient_step(self, alpha, lambda_param):
        """Gradient step."""
        # Calculate the number of training examples.
        num_examples = self.data.shape[0]

        # Predictions of hypothesis on all m examples.
        predictions = self.hypothesis(self.data, self.theta)

        # The difference between predictions and actual values for all m examples.
        delta = predictions - self.labels

        # Calculate the regularization parameter.
        reg_param = 1 - alpha * lambda_param / num_examples

        # Create the theta shortcut.
        theta = self.theta

        # Vectorized version of gradient descent.
        theta = theta * reg_param - alpha * (1 / num_examples) * (delta.T @ self.data).T
        # We should NOT regularize the parameter theta_zero.
        theta[0] = theta[0] - alpha * (1 / num_examples) * (self.data[:, 0].T @ delta).T

        self.theta = theta

    def cost_function(self, data, labels, lambda_param):
        """Cost function."""
        # Calculate the number of training examples and features.
        num_examples = data.shape[0]

        # Get the difference between predictions and correct output value.
        delta = self.hypothesis(data, self.theta) - labels

        # Calculate regularization parameter.
        # Remeber that we should not regularize the parameter theta_zero.
        theta_cut = self.theta[1:, 0]
        reg_param = lambda_param * (theta_cut.T @ theta_cut)

        # Calculate current predictions cost.
        cost = (1 / 2 * num_examples) * (delta.T @ delta + reg_param)

        # Let's extract cost value from the one and only cost numpy matrix cell.
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        """Hypothesis function."""
        return data @ theta
