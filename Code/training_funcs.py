# Cell 4: Training and Evaluation Functions

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

from sklearn.metrics import roc_auc_score, f1_score

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_probs = []
    all_labels = []

    for batch_idx, (location_data, demographics, labels) in enumerate(tqdm(train_loader, desc="Training")):

        demographics = demographics.to(device)
        labels = labels.to(device)
        
        for loc in location_data:
            for i in range(len(location_data[loc])):
                location_data[loc][i] = location_data[loc][i].to(device)

        optimizer.zero_grad()

        outputs = model(location_data, demographics)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    f1 = f1_score(all_labels, all_predictions)

    return avg_loss, accuracy, auc, f1


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on given dataset"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for location_data, demographics, labels in tqdm(test_loader, desc="Evaluating"):
            demographics = demographics.to(device)
            labels = labels.to(device)
            
            for loc in location_data:
                for i in range(len(location_data[loc])):
                    location_data[loc][i] = location_data[loc][i].to(device)

            outputs = model(location_data, demographics)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    f1 = f1_score(all_labels, all_predictions)

    return avg_loss, accuracy, auc, f1, all_predictions, all_labels

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