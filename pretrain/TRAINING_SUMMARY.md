# Training Summary: How the Model Was Trained

## Core Training Process

### 1. Data Preparation
```python
# Dataset splits: 80% train, 10% validation, 10% test
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### 2. Model Setup
```python
# ResNet-18 with modified first layer for 1-channel input (mel spectrograms)
resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet18.conv1 = nn.Conv2d(1, resnet18.conv1.out_channels, kernel_size=resnet18.conv1.kernel_size,
                           stride=resnet18.conv1.stride, padding=resnet18.conv1.padding, bias=False)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)  # Binary classification
resnet18 = resnet18.to(device)
```

### 3. Training Function
```python
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, step_size=7, gamma=0.1):
    # Learning rate scheduler (reduces LR by factor of gamma every step_size epochs)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Track best model
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # TRAINING PHASE
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            # Prepare input: add channel dimension (batch, 1, 64, time_frames)
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions.double() / total_predictions
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Update learning rate
        scheduler.step()

        # VALIDATION PHASE
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())
        
        # Save best model
        if val_acc.item() > best_val_acc:
            best_val_acc = val_acc.item()
            best_model_state = model.state_dict().copy()
            print(f"✨ New best validation accuracy: {best_val_acc:.4f}")

        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)

    loss = running_loss / len(loader.dataset)
    acc = correct_predictions.double() / total_predictions
    return loss, acc
```

### 4. Training Execution
```python
# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam with learning rate 0.001
optimizer_resnet18 = torch.optim.Adam(resnet18.parameters(), lr=0.001)

# Train for 20 epochs
train_losses, train_accuracies, val_losses, val_accuracies = train_model(
    resnet18, train_loader, val_loader, criterion, optimizer_resnet18, num_epochs=20
)
```

## Training Configuration

- **Model**: ResNet-18 (pre-trained ImageNet weights, fine-tuned)
- **Input**: Mel Spectrograms (64 mel bands, variable time frames)
- **Input Shape**: `(batch, 1, 64, time_frames)`
- **Output**: Binary classification (0=REAL, 1=FAKE)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Learning Rate Scheduler**: StepLR (step_size=7, gamma=0.1)
- **Batch Size**: 32
- **Epochs**: 20
- **Device**: CUDA (if available) or CPU

## Training Results

- **Final Training Accuracy**: ~99.8%
- **Final Validation Accuracy**: ~99.6%
- **Test Accuracy**: 99.6%
- **Best Model**: Saved based on highest validation accuracy

## Key Training Features

1. **Best Model Tracking**: Saves the model with highest validation accuracy
2. **Learning Rate Scheduling**: Reduces LR by 10x every 7 epochs
3. **Progress Tracking**: Real-time loss and accuracy display
4. **Validation Monitoring**: Evaluates on validation set after each epoch

