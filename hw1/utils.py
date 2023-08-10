import numpy as np
from simple_parsing import ArgumentParser

from args import TrainingArguments, TrainingWithVisualizationArguments
from model import LinearClassifier, LogisticRegression

def parse_args():
    parser = ArgumentParser()
    parser.add_arguments(TrainingArguments, dest='options')
    return parser.parse_args().options

def parse_visualization_args():
    parser = ArgumentParser()
    parser.add_arguments(TrainingWithVisualizationArguments, dest='options')
    return parser.parse_args().options

def load_model(name):
    return {
        'logistic_regression': LogisticRegression,
        'linear_classifier': LinearClassifier,
    }[name]

def get_model_name(args):
    return f'{args.log_dir}/{args.dataset}/num_epochs={args.num_epochs}.learning_rate={args.learning_rate}.momentum={args.momentum}'

def gen_meshgrid(grid_size, train_x, epsilon):
    assert train_x.shape[1] == 2
    x1 = train_x[:, 0]
    x1 = np.linspace(x1.min() - epsilon, x1.max() + epsilon, grid_size)
    x2 = train_x[:, 1]
    x2 = np.linspace(x2.min() - epsilon, x2.max() + epsilon, grid_size)
    xx1, xx2 = np.meshgrid(x1, x2)
    eval_x = np.concatenate((xx1.reshape(-1, 1), xx2.reshape(-1, 1)), axis=1)
    return x1, x2, eval_x