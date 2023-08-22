import datetime
import numpy as np
from simple_parsing import ArgumentParser

from args import TrainingArguments, TrainingWithVisualizationArguments
from model import LitSimpleClassifier, LitDigitsClassifier
from data import LitSimpleDataModule, LitDigitsDataModule

# Various utilities useful for training/loading/visualizing the models
# Don't add anything new here. Keep all your implementation localized to `model.py`

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
        'simple': LitSimpleClassifier,
        'digits': LitDigitsClassifier,
    }[name]

def load_datamodule(name):
    return {
        'simple': LitSimpleDataModule,
        'digits': LitDigitsDataModule,
    }[name]

def get_model_name(args):
    return f'{args.log_dir}/{args.dataset}/num_epochs={args.num_epochs}.learning_rate={args.learning_rate}.{gen_timestamp()}'

def gen_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

def gen_meshgrid(grid_size, train_x, epsilon):
    assert train_x.shape[1] == 2
    x1 = train_x[:, 0]
    x1 = np.linspace(x1.min() - epsilon, x1.max() + epsilon, grid_size)
    x2 = train_x[:, 1]
    x2 = np.linspace(x2.min() - epsilon, x2.max() + epsilon, grid_size)
    xx1, xx2 = np.meshgrid(x1, x2)
    eval_x = np.concatenate((xx1.reshape(-1, 1), xx2.reshape(-1, 1)), axis=1)
    return x1, x2, eval_x
