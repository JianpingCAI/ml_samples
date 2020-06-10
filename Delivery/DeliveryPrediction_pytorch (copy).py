import random
import numpy as np
from sklearn import preprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from sklearn.metrics import mean_squared_error
from numpy import sort
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from NeuralNetRegressor import *
from DeliveryDataLoader import *

seed_all(1234)

# load data files
train_dataset = pd.read_excel('data_train.xlsx')
test_dataset = pd.read_excel('data_test.xlsx')

train_dataset.shape
train_dataset.describe()
train_dataset.head()

# preprocess train & test data
X_train_raw, y_train = data_preprocess(train_dataset)
X_test_raw, y_test = data_preprocess(test_dataset)
encoders = encode_categorical_columns(X_train_raw, X_test_raw)


test = DatasetDelivery(X_train_raw, y_train)
test.__getitem__(0)

input_dim = len(X_train_raw.columns)

# MinMaxScaler
# mm_scaler = preprocessing.MaxAbsScaler()
mm_scaler = preprocessing.MinMaxScaler()
X_train_scale = pd.DataFrame(mm_scaler.fit_transform(
    X_train_raw), columns=X_train_raw.columns)
X_test_scale = pd.DataFrame(mm_scaler.transform(
    X_test_raw), columns=X_test_raw.columns)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_scale, y_train, test_size=0.2, random_state=42)

# batch split
num_splits = 50
train_batches = np.array_split(X_train, num_splits)
targets_batches = np.array_split(y_train, num_splits)
print("batch_size:{}".format(len(train_batches[0].index)))

# convert to tensors
for i in range(len(train_batches)):
    train_batches[i] = torch.from_numpy(train_batches[i].values).float()
for i in range(len(targets_batches)):
    targets_batches[i] = torch.from_numpy(
        targets_batches[i].values).float().view(-1, 1)
X_val = torch.from_numpy(X_val.values).float()
y_val = torch.from_numpy(y_val.values).float().view(-1, 1)


# hyperparameter to penalize more on underestimation cases (if less than 0)
hyper_a = -0.2


def delivery_loss_func(output, target):
    '''
    FC NNet Regressor
        higher loss for under-estimation if hyper_a < 0
    '''
    loss = torch.mean(((output - target)**2) *
                      ((torch.sign(output-target)+hyper_a)**2))
    return loss


# ML model
model = MLPRegressor(input_dim)
# criterion = nn.MSELoss()
criterion = delivery_loss_func
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 100
train_losses, val_losses = [], []
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for i in range(len(train_batches)):
        print(i)
        # Forward pass
        outputs = model(train_batches[i])
        # loss = torch.sqrt(criterion(outputs, targets_batch[i]))
        loss = delivery_loss_func(outputs, targets_batches[i])

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    val_loss = 0
    with torch.no_grad():
        model.eval()
        predictions = model(X_val)
        # val_loss = torch.sqrt(criterion(predictions, y_val))
        val_loss = delivery_loss_func(predictions, y_val)

    train_losses.append(train_loss/len(train_batches))
    val_losses.append(val_loss)

    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
          "Training Loss: {:.3f}.. ".format(train_loss/len(train_batches)),
          "Val Loss: {:.3f}.. ".format(val_loss))

pyplot.plot(train_losses, label='Training loss')
pyplot.plot(val_losses, label='Validation loss')
pyplot.legend(frameon=False)


# inference on test dataset
test = torch.from_numpy(X_test_scale.values).float()

with torch.no_grad():
    model.eval()
    output = model.forward(test)

# statistics
predictions = output.numpy()
predictions = [round(value) for value in predictions.flatten()]
mse = mean_squared_error(y_test.values, predictions, squared=True)
print("test mse: %.2f" % (mse))

diffs = predictions-y_test.values
num_underestimate = np.sum(diffs < 0)
num_overestimate = np.sum(diffs > 0)
num_correct = np.sum(diffs == 0)
print("under-estimated:   {}, {:.2f}%".format(num_underestimate,
                                              num_underestimate/len(diffs)*100))
print("over-estimated :   {}, {:.2f}%".format(num_overestimate,
                                              num_overestimate/len(diffs)*100))
print("correct-estimated: {}, {:.2f}%".format(num_correct, num_correct/len(diffs)*100))

# Save the model checkpoint
torch.save(model.state_dict(), 'pytorch_model.ckpt')

# y_test = torch.from_numpy(y_test.values).float().view(-1, 1)
# mse_test = torch.sqrt(criterion(output, y_test)).item()
# mse_test = nn.MSELoss()(output, y_test).item()
# print('mse %.2f' % mse_test)
