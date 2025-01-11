import torch
import torch.nn as nn
import torch.utils
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# hyper parameters
input_size = 784  # 28 x 28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
lr = 0.001

#MNIST
train_dataset = torchvision.datasets.MNIST(root = './/data', train= True, transform= transforms.ToTensor(), download= True)

test_dataset = torchvision.datasets.MNIST(root = './/data', train= False, transform= transforms.ToTensor(), download= False)

train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle = False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()    

class NeuralNet(nn.Module): 
    def __init__(self, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # (1 input channel, 32 filters)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce spatial size by half
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # (32 input channels, 64 filters)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)  # Flatten from 7x7x64 after pooling
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Pass through convolutional and pooling layers
        x = self.pool(F.relu(self.conv1(x)))  # Output size: 14x14x32
        x = self.pool(F.relu(self.conv2(x)))  # Output size: 7x7x64
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Output layer
        return x
  
model = NeuralNet(hidden_size, num_classes).to(device)

#loss adnoptimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= lr)

#training loop

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (image, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784
  
        labels = labels.to(device)
        image = image.to(device)
        # forward
        outputs = model(image.to(device))
        loss = criterion(outputs, labels)


        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 ==0:
            print(f'epoch: {epoch+1} / {num_epochs}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')

# test 
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for image, labels in test_loader:
        image = image.to(device)
        labels = labels.to(device)
        outputs = model(image)

        # Predictions
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f'accuracy = {acc:.2f} %')

