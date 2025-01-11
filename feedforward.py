import torch
import torch.nn as nn
import torch.utils
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

#loss adnoptimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= lr)

#training loop
import time

# Function to train and test the model
def train_and_test(device):
    # Move model to the selected device
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        for i, (image, labels) in enumerate(train_loader):
            # Move data to the selected device
            image = image.reshape(-1, 784).to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(image)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time on {device}: {training_time:.2f} seconds")

    # Test the model
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for image, labels in test_loader:
            image = image.reshape(-1, 784).to(device)
            labels = labels.to(device)
            outputs = model(image)

            # Predicted classes
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy on {device}: {acc:.2f} %")

# Measure time for CUDA
if torch.cuda.is_available():
    print("Running on CUDA:")
    train_and_test(torch.device('cuda'))

# Measure time for CPU
print("\nRunning on CPU:")
train_and_test(torch.device('cpu'))

'''
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (image, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784
        image = image.reshape(-1,784).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(image)
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
        image = image.reshape(-1,784).to(device)
        labels = labels.to(device)
        outputs = model(image)

        # value, index
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct += (predictions==labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f'accuracy = {acc} %')


'''
