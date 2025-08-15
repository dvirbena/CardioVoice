
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

