from __future__ import print_function
import enum
from matplotlib.cbook import ls_mapper
from matplotlib.transforms import Transform 
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import sys
import torch
sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation
from sklearn.utils.class_weight import compute_class_weight

def read_data(datasets_dir="./data", frac=0.3):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=0):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    X_train, X_valid = rgb2gray(X_train), rgb2gray(X_valid)
    X_train, X_valid = [x[12:-12, 12:-12] for x in X_train], [x[12:-12, 12:-12] for x in X_valid]
    X_train = [X_train[i-history_length-1:i] for i in range(history_length+1, len(X_train))]
    X_valid = [X_valid[i-history_length-1:i] for i in range(history_length+1, len(X_valid))]
    y_train = [action_to_id(y) for y in y_train][history_length+1:]
    y_valid = [action_to_id(y) for y in y_valid][history_length+1:]

    return torch.Tensor(np.array(X_train)), torch.Tensor(np.array(y_train)), torch.Tensor(np.array(X_valid)), torch.Tensor(np.array(y_valid))


def train_model(train, X_train, y_train, valid, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard", history_length=1):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    device = torch.device('cuda')

    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    class_weights = torch.Tensor(np.bincount(y_train) / len(y_train))
    class_weights = class_weights.to(device)
    sample_weights = torch.Tensor([1/class_weights[int(y)] for y in y_train])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train))
    
    agent = BCAgent(history_length=history_length, class_weights=class_weights)
    
    tensorboard_eval = Evaluation(tensorboard_dir, 'imitation_learning', ['train-loss', 'train-acc', 'valid-loss', 'valid-acc'])

    # TODO: implement the training

    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser

    # training loop
    train_loader = DataLoader(train, batch_size=batch_size, sampler=sampler)
    transform = T.Compose([
       T.ToTensor()
    ])

    train_loss, train_acc, valid_loss, valid_acc = 0, 0, 0, 0
    for i in range(n_minibatches):
        X_batch, y_batch = next(iter(train_loader))
        X_batch = transform(X_batch)
        y_batch = y_batch.type(torch.LongTensor)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        loss, logits = agent.update(X_batch, y_batch)
        train_loss += loss
        train_acc += accuracy(logits, y_batch)

        with torch.no_grad():
            y_valid = y_valid.type(torch.LongTensor)
            X_valid, y_valid = X_valid.to(device), y_valid.to(device)
            logits = agent.predict(X_valid)
            valid_loss += agent.loss_fn(logits, y_valid)
            valid_acc += accuracy(logits, y_valid)

        if i % 10 == 0:
            # TODO: compute training/ validation accuracy and write it to tensorboard
            print('minibatch: ', i, ' train-loss: ', float(train_loss/10), ' train-acc: ', float(train_acc/10), ' val_loss: ', float(valid_loss/10), ' valid-acc: ', float(valid_acc/10))
            tensorboard_eval.write_episode_data(i, {'train-loss': train_loss/10, 'train-acc': train_acc/10, 'valid-loss': valid_loss/10, 'valid-acc': valid_acc/10})
            train_loss, train_acc, valid_loss, valid_acc = 0, 0, 0, 0

    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)


def accuracy(logits, y):
    y_pred = torch.argmax(logits, axis=1)

    return torch.sum(y_pred == y) / len(y)


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=0)

    train, valid = TensorDataset(X_train, y_train), TensorDataset(X_valid, y_valid)

    # train model (you can change the parameters!)
    train_model(train, X_train, y_train, valid, X_valid, y_valid, n_minibatches=10000, batch_size=64, lr=1e-4, history_length=0)
 
