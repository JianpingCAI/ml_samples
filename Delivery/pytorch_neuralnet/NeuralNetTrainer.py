import torch
from sklearn.metrics import mean_squared_error
import numpy as np


class MLPTrainer():
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        pass

    def delivery_loss_func(self, output, target, hyper_a):
        '''
        FC NNet Regressor
            higher loss for under-estimation if hyper_a < 0
        '''
        diffs = output - target

        mse = torch.mean((diffs**2))
        panalty = torch.mean((torch.sign(diffs)+hyper_a)**2)

        loss = mse * panalty
        return loss

    def train(self, train_loader, val_loader, hyper_a=-0.2, num_epochs=200):
        epochs = num_epochs
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            # for i in range(len(train_batch)):
            for i, (batch, targets) in enumerate(train_loader):
                # Forward pass
                outputs = self.model(batch)
                loss = self.delivery_loss_func(outputs, targets, hyper_a)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            val_loss = 0
            with torch.no_grad():
                self.model.eval()
                for i, (batch, targets) in enumerate(val_loader):
                    # Forward pass
                    outputs = self.model(batch)
                    loss = self.delivery_loss_func(outputs, targets, hyper_a)

                    val_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Val Loss: {:.3f}.. ".format(val_loss))

        return train_losses, val_losses

    def online_train(self, train_loader, val_loader, hyper_a=-0.2, num_epochs=200):
        pass