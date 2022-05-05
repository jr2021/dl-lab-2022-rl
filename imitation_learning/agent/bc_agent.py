import torch
from agent.networks import CNN
import torch.nn as nn
import torch.nn.functional as F

class BCAgent:
    
    def __init__(self, class_weights=[0.25, 0.25, 0.25, 0.25], history_length=0):
        # TODO: Define network, loss function, optimizer
        self.model = CNN(history_length=history_length)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def update(self, X_batch, y_batch):
        # TODO: forward + backward + optimize
        self.optimizer.zero_grad()
        y_pred = self.predict(X_batch)
        loss = self.loss_fn(y_pred, torch.LongTensor(y_batch))
        loss.backward()
        self.optimizer.step()

        return loss, y_pred

    def predict(self, X):
        # TODO: forward pass
        X = torch.Tensor(X)
        outputs = self.model(X)

        return outputs


    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))


    def save(self, file_name):
        torch.save(self.model.state_dict(), file_name)

