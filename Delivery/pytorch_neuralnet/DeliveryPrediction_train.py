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
from NeuralNetTrainer import *

seed_all(1234)

# load data files
train_dataset = pd.read_excel('../data_train.xlsx')
test_dataset = pd.read_excel('../data_test.xlsx')

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

# dataloader
batch_size = 32
train_dataset = DatasetDelivery(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

batch_size_val = len(X_val.index)
test_dataset = DatasetDelivery(X_val, y_val)
val_loader = DataLoader(
    test_dataset, batch_size=batch_size_val, shuffle=False)


# ML model
model = MLPRegressor(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
#val_criterion = nn.MSELoss()

# train
# a hyperparameter to penalize more on underestimation cases (if < 0)
hyper_a = -0.2
trainer = MLPTrainer(model, optimizer)
train_losses, val_losses = trainer.train(
    train_loader, val_loader, hyper_a, num_epochs=200)

# plot losses
pyplot.plot(train_losses, label='Training loss')
pyplot.plot(val_losses, label='Validation loss')
pyplot.legend(frameon=False)

# inference on test dataset
model_inference(model, X_test_scale, y_test)

# Save the model checkpoint
torch.save(model.state_dict(), 'pytorch_model.ckpt')
