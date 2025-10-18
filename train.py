from model import SignalClassifier
from dataset import train_loader, test_loader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# initialize model, loss, and optimizer
model = SignalClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_history = []
accuracy_history = []
num_epochs = 20

for epoch in range(num_epochs):
    # training phase
    model.train()
    total_loss = 0

    for signals, labels in train_loader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    # evaluation phase
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    accuracy_history.append(accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

plt.figure()
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

plt.figure()
plt.plot(accuracy_history, label='Test Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Over Time')
plt.legend()
plt.show()