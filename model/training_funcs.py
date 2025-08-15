def custom_collate_fn(batch):
    """Custom collate function to handle variable number of clips per location"""
    location_data = {loc: [] for loc in ['AV', 'PV', 'TV', 'MV']}
    demographics = []
    labels = []
    
    for location_dict, demo, label in batch:
        # Collect demographics and labels
        demographics.append(demo)
        labels.append(label)
        
        # Collect location data
        for loc in ['AV', 'PV', 'TV', 'MV']:
            location_data[loc].append(location_dict[loc])
    
    # Stack tensors
    demographics_tensor = torch.stack(demographics)
    labels_tensor = torch.LongTensor(labels)
    
    return location_data, demographics_tensor, labels_tensor

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0


    for batch_idx, (location_data, demographics, labels) in enumerate(tqdm(train_loader, desc="Training")):

        # Move data to device
        demographics = demographics.to(device)
        labels = labels.to(device)
        
        # Move location data to device
        for loc in location_data:
            for i in range(len(location_data[loc])):
                location_data[loc][i] = location_data[loc][i].to(device)

        optimizer.zero_grad()


        # Forward pass
        outputs = model(location_data, demographics)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for location_data, demographics, labels in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            demographics = demographics.to(device)
            labels = labels.to(device)
            
            # Move location data to device
            for loc in location_data:
                for i in range(len(location_data[loc])):
                    location_data[loc][i] = location_data[loc][i].to(device)

            
            
            # Forward pass
            outputs = model(location_data, demographics)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Store predictions and labels for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy, all_predictions, all_labels

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Abnormal']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()#, attn_weights
