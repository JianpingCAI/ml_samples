import random
import numpy as np
from sklearn import preprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import torchvision
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

# preprocess train & test data
X_train_raw, y_train_raw = data_preprocess(train_dataset)
X_test_raw, y_test = data_preprocess(test_dataset)
encoders = encode_categorical_columns(X_train_raw, X_test_raw)

input_dim = len(X_train_raw.columns)

# MinMaxScaler
mm_scaler = preprocessing.MinMaxScaler()  # MaxAbsScaler()
X_train_scale = pd.DataFrame(mm_scaler.fit_transform(
    X_train_raw), columns=X_train_raw.columns)
X_test_scale = pd.DataFrame(mm_scaler.transform(
    X_test_raw), columns=X_test_raw.columns)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_scale, y_train_raw, test_size=0.2, random_state=42)

batch_size = 32
train_dataset = DatasetDelivery(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# hyperparameter to penalize more on underestimation cases (if less than 0)
hyper_a = -0.2


def delivery_loss_func(output, target):
    '''
    FC NNet Regressor
        higher loss for under-estimation if hyper_a < 0
    '''
    diffs = output - target

    mse = torch.mean((diffs**2))
    panalty = torch.mean((torch.sign(diffs)+hyper_a)**2)

    loss = mse * panalty
    return loss


# ML model
model = MLPRegressor(input_dim)
val_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# convert to tensors
X_val = torch.from_numpy(X_val.values.astype(np.float32))
y_val = torch.from_numpy(y_val.values.astype(np.float32)).view(-1, 1)

epochs = 200
train_losses, val_losses = [], []
for epoch in range(epochs):
    model.train()
    train_loss = 0
    # for i in range(len(train_batch)):
    for i, (batch, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(batch)
        loss = delivery_loss_func(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    val_loss = 0
    with torch.no_grad():
        model.eval()
        predictions = model(X_val)
        #val_loss = val_criterion(predictions, y_val)
        val_loss = delivery_loss_func(predictions, y_val)

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
          "Training Loss: {:.3f}.. ".format(train_loss),
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
