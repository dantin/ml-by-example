# -*- coding: utf-8 -*-
import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd

import homemade.regression.linear as regs


logger = logging.getLogger('demo')
default_handler = logging.StreamHandler()
default_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
logger.addHandler(default_handler)


def plot_dataframe_histogram(df, path, suffix='png'):
    """Plot dataframe histogram."""
    try:
        histograms = df.hist(grid=False, figsize=(10, 10))
    except Exception as e: # pylint: disable=broad-exception
        logger.warning('plot error, exceptions: %s', e)
    else:
        plt.savefig(path, format=suffix)
    finally:
        plt.close('all')


def plot_training_data(data, path, suffix='png'):
    """Plot training data."""
    try:
        x_train, y_train  = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
        x_label, y_label = data['x_label'], data['y_label']
        title = data['title']

        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(111)

        plt.scatter(x_train, y_train, label='Training Dataset')
        plt.scatter(x_test, y_test, label='Test Dataset')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
    except KeyError as e:
        logger.warning('key not found, exceptions: %s', e)
    except Exception as e:
        logger.warning('plot error, exceptions: %s', e)
    else:
        plt.savefig(path, format=suffix)
    finally:
        plt.close('all')


def plot_gradient_descent_progess(num_iterations, cost_history, path, suffix='png'):
    """Plot gradient descent progress."""
    try:
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(111)

        plt.plot(range(num_iterations), cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Gradient Descent Progress')
    except Exception as e:
        logger.warning('plot gradient descent progress error, exceptions: %s', e)
    else:
        plt.savefig(path, format=suffix)
    finally:
        plt.close('all')


def load_data(path):
    """Load corpus data from CSV file."""
    # Load the data.
    data = pd.read_csv(path)

    # Print the data table.
    logger.debug('corpus header:\n%s', data.head(10))
    return data


def split_data(data, input_param_name, output_param_name):
    """Split the data into training and test subsets."""
    # Split dataset on training and test sets with proportions 80/20.
    # Method `sample()` returns a random sample of items.
    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)

    # Show fields we want to process.
    logger.debug(' Input param name:  %s', input_param_name)
    logger.debug(' Output param name: %s', output_param_name)

    # Split training set input and output.
    x_train = train_data[[input_param_name]].values
    y_train = train_data[[output_param_name]].values

    # Split test set input and output.
    x_test = test_data[[input_param_name]].values
    y_test = test_data[[output_param_name]].values

    return x_train, y_train, x_test, y_test


def train(x_train, y_train):
    """Training process."""
    num_iterations = 500     # Number of gradient descent iterations.
    regularization_param = 0 # Helpers to fight model overfitting.
    learning_rate = 0.01     # The size of the gradient descent step.
    polynomial_degree = 0    # The degree of additional polynomial features.
    sinusoid_degree = 0      # The degree of sinusoid parameter multipliers of additional features.

    # Init linear regression instance.
    linear_reg = regs.LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
    
    theta, cost_history = linear_reg.train(learning_rate, regularization_param, num_iterations)

    logger.info(' Initial cost:   %.2f', cost_history[0])
    logger.info(' Optimized cost: %.2f', cost_history[-1])

    theta_table = pd.DataFrame({'Model Parameters': theta.flatten()})
    logger.debug('Model parameters:\n%s', theta_table.head())

    logger.debug('Plot gradient descent progress.')
    plot_gradient_descent_progess(num_iterations, cost_history, 'gradient_descent_progress.png')


def main():
    """Bootstrap function."""
    parser = argparse.ArgumentParser(description='Demo of Univariate Linear Regression')
    parser.add_argument('--corpus', dest='corpus_file', default='data/world-happiness-report-2017.csv', help='corpus file')
    parser.add_argument('--input', dest='input_param', default='Economy..GDP.per.Capita.', help='input parameter')
    parser.add_argument('--predict', dest='predict_param', default='Happiness.Score', help='prediction parameter')
    parser.add_argument('--title', dest='title', default='Countries Happiness', help='default figure title')
    args = parser.parse_args()

    # Load data
    data = load_data(args.corpus_file)
    logger.debug('plot histograms for each feature to see how they vary')
    plot_dataframe_histogram(data, 'raw_features.png')

    # Split data
    x_train, y_train, x_test, y_test = split_data(data, args.input_param, args.predict_param)
    plot_data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'title': args.title,
        'x_label': args.input_param,
        'y_label': args.predict_param,
    }
    plot_training_data(plot_data, 'training_data.png')

    logger.info('train process')
    train(x_train, y_train)


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    main()
