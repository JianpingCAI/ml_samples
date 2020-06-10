import torch as t
import torch.version
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision

x = t.tensor(1., requires_grad=True)
w = t.tensor(2., requires_grad=True)
b = t.tensor(3., requires_grad=True)

y = w*x+b
y.backward()

print(x.grad.item())
print(w.grad)
print(b.grad)

x = t.randn(10, 3)
y = t.randn(10, 2)
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b:', linear.bias)

criterion = nn.MSELoss()
optimizer = t.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)

loss = criterion(pred, y)
print('loss: ', loss.item())

loss.backward()

print('dL/dw', linear.weight.grad)
print('dL/db', linear.bias.grad)

optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
print('loss', loss.item())


x = np.array([[1, 2], [3, 4]])
y = torch.from_numpy(x)
z = y.numpy()

train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

image, label = train_dataset[0]
print(image.size())
print(label)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass