# Cell 7: Usage Example

def predict_single_patient(model, preprocessor, demographic_scaler, patient_data, device):
    """Make prediction for a single patient"""
    model.eval()
    
    # Create dataset with single patient
    dataset = HeartSoundDataset([patient_data], preprocessor, demographic_scaler)
    
    if len(dataset) == 0:
        return None, "No valid data for prediction"
    
    # Get data
    location_data, demographics, _ = dataset[0]
    
    # Add batch dimension and move to device
    demographics = demographics.unsqueeze(0).to(device)
    
    # Prepare location data
    batch_location_data = {}
    for loc in ['AV', 'PV', 'TV', 'MV']:
        batch_location_data[loc] = [location_data[loc].to(device)]
    
    with torch.no_grad():
        outputs = model(batch_location_data, demographics)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    prediction = "Normal" if predicted.item() == 0 else "Abnormal"
    confidence = probabilities[0][predicted.item()].item()
    
    return prediction, confidence

# Example usage and training script
if __name__ == "__main__":
    # Set your data directory path here
    DATA_DIR = "C:\\Users\\Yosss\\Desktop\\the-circor-digiscope-phonocardiogram-dataset-1.0.3\\the-circor-digiscope-phonocardiogram-dataset-1.0.3\\training_data"
    
    # Check if directory exists (comment out if you want to skip this check)
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found!")
        print("Please update DATA_DIR variable with the correct path to your CirCor dataset")
        print("\nTo use this code:")
        print("1. Download the CirCor dataset from PhysioNet")
        print("2. Extract it to a directory")
        print("3. Update the DATA_DIR variable above")
        print("4. Run the training pipeline")
        exit()
    
    # Run training pipeline
    try:
        trained_model, preprocessor, scaler, viz = main_training_pipeline(
            data_dir=DATA_DIR,
            batch_size=32,  # Adjust based on your GPU memory
            num_epochs=100,
            learning_rate=0.0001
        )
        
        print("Training completed successfully!")
        print("Model saved as 'best_heart_sound_model.pth'")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()