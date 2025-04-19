import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
batch_size = 32
learning_rate = 0.1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./MNIST', 
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./MNIST',
    train=False,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)  
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 10) 
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.softmax(x)
        return x

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate) 
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.99)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f'Test Accuracy: {100 * correct / total:.2f}%')


torch.save(model.state_dict(), 'mnist_cnn.pth')

def save_model(model, path):
    param_list = []
    param_list.append(model.conv1.weight.detach().tolist())
    param_list.append(model.conv2.weight.detach().tolist())
    
    with open(path, 'wb') as f:
        pickle.dump(param_list, f)


save_model(model, r'C:\Users\wzzj1\Downloads\PJ1\codes\saved_models\base_cnn.pickle')


model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth'))


conv1_weights = model.conv1.weight.detach().cpu()

fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 2
for i in range(columns * rows):
    kernel = conv1_weights[i][0]  
    ax = fig.add_subplot(rows, columns, i+1)
    ax.imshow(kernel, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Kernel {i+1}')

plt.suptitle('First Conv Layer Kernels')
plt.show()


conv2_weights = model.conv2.weight.detach().cpu()

fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 4
for i in range(columns * rows):
    kernel = conv2_weights[i][0]  
    ax = fig.add_subplot(rows, columns, i+1)
    ax.imshow(kernel, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Kernel {i+1}')

plt.suptitle('Second Conv Layer Kernels')
plt.show()