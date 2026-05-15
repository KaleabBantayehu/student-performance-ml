import pandas as pd
import numpy as np
from src.preprocess import DataPreprocessor
from sklearn.model_selection import train_test_split

def test_pipeline():
    """
    Validates the preprocessing pipeline and checks data quality.
    """
    print("\n" + "="*60)
    print(" SECTION 1: PREPROCESSING TEST & QUALITY CHECK ".center(60))
    print("="*60)
    
    data_file = "data/student_data.csv"
    processor = DataPreprocessor(data_path=data_file)
    
    try:
        X, y, feature_names = processor.process()
        
        print(f"[OK] Dataset loaded successfully")
        print(f"Features shape (X): {X.shape}")
        print(f"Labels shape (y): {y.shape}")
        
        # --- PREPROCESSING VALIDATION LOGIC ---
        print("\n--- Data Integrity & Encoding Check ---")
        
        # Check categorical encoding (assignments_completed is usually index 3)
        print(f"First 5 rows of 'assignments_completed' (encoded): {X[:5, 3]}")
        
        # Check for NaN values
        nan_count = np.isnan(X).sum()
        print(f"Total NaN values in features: {nan_count}")
        
        # Check label encoding mapping
        mapping = dict(zip(processor.label_encoder.classes_, range(len(processor.label_encoder.classes_))))
        print(f"Label Mapping: {mapping}")
        print(f"First 5 encoded labels: {y[:5]}")

    except Exception as e:
        print(f"\n[ERROR] Preprocessing Pipeline failed: {e}")
        raise e

def main():
    """
    Main execution flow for data preparation and splitting.
    """
    print("\n" + "="*60)
    print(" SECTION 2: DATA PREPARATION & SPLITTING ".center(60))
    print("="*60)
    
    data_file = "data/student_data.csv"
    processor = DataPreprocessor(data_path=data_file)
    
    try:
        # 1. Preprocess
        X, y, feature_names = processor.process()

        # 2. Split the data
        # test_size=0.2 means 20% for testing, 80% for training
        # random_state=42 ensures reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Total samples processed: {len(X)}")
        print(f"Training set size:      {X_train.shape[0]}")
        print(f"Testing set size:       {X_test.shape[0]}")
        print("\n[SUCCESS] System initialization complete. Ready for model training.")
        
    except Exception as e:
        print(f"\n[ERROR] Main execution flow failed: {e}")

if __name__ == "__main__":
    # Execution Flow: Run test first, then main
    test_pipeline()
    main()