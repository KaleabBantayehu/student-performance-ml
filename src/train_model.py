import os
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StudentModelTrainer:
    """
    A production-style trainer for the Student Performance Prediction System.
    Handles data splitting, model training, and persistence.
    """

    def __init__(self, model_path: str = "models/student_performance_model.pkl"):
        """
        Initializes the trainer with a path to save the model.
        
        Args:
            model_path (str): The destination path for the saved model.
        """
        self.model_path = model_path
        self.model = None
        self.encoder = None
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def train(self, X, y, encoder=None, test_size: float = 0.2, random_state: int = 42):
        """
        Splits the data, trains a RandomForestClassifier, and evaluates performance.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Label vector.
            encoder (LabelEncoder, optional): The encoder used for target labels.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before splitting.
            
        Returns:
            tuple: (trained_model, X_test, y_test)
            
        Raises:
            Exception: If training fails.
        """
        try:
            self.encoder = encoder
            logger.info("Splitting data into training and testing sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            logger.info(f"Initializing RandomForestClassifier with random_state={random_state}...")
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)

            logger.info("Starting model training...")
            self.model.fit(X_train, y_train)
            
            # Simple evaluation for logging purposes
            train_acc = accuracy_score(y_train, self.model.predict(X_train))
            test_acc = accuracy_score(y_test, self.model.predict(X_test))
            
            logger.info(f"Training successful. Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            # Save artifacts automatically after training
            self.save_model()
            
            return self.model, X_test, y_test

        except Exception as e:
            logger.error(f"An error occurred during model training: {str(e)}")
            raise

    def save_model(self, encoder_path: str = "models/label_encoder.pkl"):
        """
        Persists the trained model and label encoder to disk using joblib.
        
        Args:
            encoder_path (str): The destination path for the label encoder.
            
        Raises:
            ValueError: If no model has been trained yet.
            Exception: If saving fails.
        """
        if self.model is None:
            raise ValueError("No model trained. Call train() before saving.")
            
        try:
            logger.info(f"Saving model to {self.model_path}...")
            joblib.dump(self.model, self.model_path)
            
            if self.encoder is not None:
                logger.info(f"Saving label encoder to {encoder_path}...")
                joblib.dump(self.encoder, encoder_path)
            else:
                logger.warning("No label encoder provided. Skipping encoder persistence.")
                
            logger.info("Artifacts saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save the model: {str(e)}")
            raise

if __name__ == "__main__":
    # This block is for independent testing of the module
    from src.preprocess import DataPreprocessor
    
    try:
        # Load and preprocess data
        preprocessor = DataPreprocessor(data_path="data/student_data.csv")
        X, y, _ = preprocessor.process()
        
        # Initialize and run trainer
        trainer = StudentModelTrainer()
        model, X_test, y_test = trainer.train(X, y, encoder=preprocessor.label_encoder)
        
        print("\n--- Training Pipeline Verification ---")
        print(f"Model saved at: {trainer.model_path}")
        print(f"Test samples: {len(X_test)}")
        
    except Exception as e:
        print(f"Verification failed: {e}")
