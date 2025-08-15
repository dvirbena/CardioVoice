# Cell 7: Usage Example

# Example usage and training script
if __name__ == "__main__":
    DATA_DIR = ".\\training_data" #SET YOUR DATA DIRECTORY HERE
    
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
