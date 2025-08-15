
# Cell 5: Main Training Pipeline

def main_training_pipeline(data_dir, batch_size=8, num_epochs=50, learning_rate=0.001):
    """Complete training pipeline"""
    
    print("Loading dataset...")
    patients_data = load_dataset(data_dir)
    print(f"Loaded {len(patients_data)} patients")
    
    # Split data into train, validation, and test sets
    train_data, temp_data = train_test_split(patients_data, test_size=0.4, random_state=42, 
                                           stratify=[p.get('outcome', 'Unknown') for p in patients_data])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42,
                                         stratify=[p.get('outcome', 'Unknown') for p in temp_data])
    
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(normalization_method='zscore')
    
    # Prepare demographic scaler
    all_demographics = []
    for patient in train_data:
        if patient.get('outcome') in ['Normal', 'Abnormal']:
            # Extract demographic features for scaling
            features = []
            features.append(float(patient.get('height', 120.0)) if patient.get('height') else 120.0)
            features.append(float(patient.get('weight', 25.0)) if patient.get('weight') else 25.0)
            features.append(1 if patient.get('sex') == 'Male' else 0)
            age_mapping = {'Neonate': 0, 'Infant': 1, 'Child': 2, 'Adolescent': 3, 'Young Adult': 4}
            features.append(age_mapping.get(patient.get('age'), 2))
            features.append(1 if patient.get('pregnancy_status') == 'True' else 0)
            all_demographics.append(features)
    
    demographic_scaler = StandardScaler()
    demographic_scaler.fit(all_demographics)
    
    # Create datasets
    train_dataset = HeartSoundDataset(train_data, preprocessor, demographic_scaler)
    val_dataset = HeartSoundDataset(val_data, preprocessor, demographic_scaler)
    test_dataset = HeartSoundDataset(test_data, preprocessor, demographic_scaler)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=custom_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          collate_fn=custom_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=custom_collate_fn, num_workers=0)

    # Initialize model
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = HeartSoundClassifier(embedding_dim=128, num_classes=2)
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    patience_counter = 0
    patience_limit = 10
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_predictions, val_labels = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(val_labels, val_predictions, 
                                target_names=['Normal', 'Abnormal']))
        
        # Early stopping
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_heart_sound_model.pth')
            print(f"New best validation accuracy: {best_val_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_heart_sound_model.pth'))
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_predictions, test_labels = evaluate_model(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions, 
                              target_names=['Normal', 'Abnormal']))
    
    # Plot results
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
    plot_confusion_matrix(test_labels, test_predictions)
    viz = quick_visualization_setup(model, device, test_loader)

    return model, preprocessor, demographic_scaler, viz


def plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Abnormal']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()#, attn_weights

class HeartSoundClassifier(nn.Module):
    """Complete heart sound classification model"""
    
    def __init__(self, embedding_dim=128, num_classes=2):
        super(HeartSoundClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.locations = ['AV', 'PV', 'TV', 'MV']
        
        # CNN for processing audio clips
        self.audio_cnn = AudioCNN(embedding_dim=embedding_dim)
        
        # MLP for processing demographics
        self.demographic_mlp = DemographicMLP(embedding_dim=embedding_dim)
        
        # Attention mechanism
        self.attention = MultiHeadAttention(embed_dim=embedding_dim)
        
        # Final classification layers
        # 5 embeddings (4 locations + demographics) * embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(5 * embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5/2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5/2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, location_data, demographics):
        batch_size = demographics.size(0)
        location_embeddings = []
        
        # Process each location
        for location in self.locations:
            location_clips = location_data[location]  # Shape: (batch_size, num_clips, channels, height, width)
            
            # Process all clips for this location in the batch
            batch_location_embeddings = []
            
            for b in range(batch_size):
                clips = location_clips[b]  # Shape: (num_clips, channels, height, width)
                
                if clips.size(0) > 0:
                    # Process all clips through CNN
                    clip_embeddings = self.audio_cnn(clips)  # Shape: (num_clips, embedding_dim)
                    # Take mean across clips to get location embedding
                    location_embedding = clip_embeddings.mean(dim=0)  # Shape: (embedding_dim,) #YOSSI CHANGE
                    #location_embedding, _ = torch.max(clip_embeddings, dim=0) #CHAT SUGGESTIONS
                    #print(f"first 5 {clip_embeddings[:,0:5]}, mean: {location_embedding[0:5]}")
                    #location_embedding = clip_embeddings[0]  # Shape: (embedding_dim,)
                else:
                    # Handle case where no clips are available
                    location_embedding = torch.zeros(self.embedding_dim, device=demographics.device)
                
                batch_location_embeddings.append(location_embedding)
            
            # Stack embeddings for all samples in batch
            batch_location_tensor = torch.stack(batch_location_embeddings)  # Shape: (batch_size, embedding_dim)
            location_embeddings.append(batch_location_tensor)
        
        # Process demographics
        demographic_embeddings = self.demographic_mlp(demographics)  # Shape: (batch_size, embedding_dim)
        
        # Combine all embeddings
        all_embeddings = location_embeddings + [demographic_embeddings]
        combined_embeddings = torch.stack(all_embeddings, dim=1)  # Shape: (batch_size, 5, embedding_dim)
        
        # Apply attention
        attended_embeddings, attention_weights = self.attention(combined_embeddings)
        
        # Flatten for final classification
        flattened = attended_embeddings.view(batch_size, -1)  # Shape: (batch_size, 5 * embedding_dim)
        
        # Final classification
        output = self.classifier(flattened)
        
        return output