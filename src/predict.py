import os
import logging
import joblib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformancePredictor:
    """
    A modular prediction component for the Student Performance Prediction System.
    Handles loading trained artifacts and performing inference on new student data.
    """
    
    def __init__(self, model_path: str = "models/student_performance_model.pkl", 
                 encoder_path: str = "models/label_encoder.pkl"):
        """
        Initializes the predictor and loads the necessary ML artifacts.
        
        Args:
            model_path (str): Path to the saved RandomForest model.
            encoder_path (str): Path to the saved LabelEncoder.
        """
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.encoder = None
        
        # Load artifacts immediately upon initialization
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Loads the trained model and label encoder from disk using joblib.
        
        Raises:
            FileNotFoundError: If any of the artifact files are missing.
            Exception: If artifact loading fails for other reasons.
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            if not os.path.exists(self.encoder_path):
                raise FileNotFoundError(f"Encoder file not found at {self.encoder_path}")
            
            logger.info("Loading prediction artifacts...")
            self.model = joblib.load(self.model_path)
            self.encoder = joblib.load(self.encoder_path)
            logger.info("Prediction artifacts successfully loaded.")
            
        except Exception as e:
            logger.error(f"Critical error loading artifacts: {str(e)}")
            raise

    def predict(self, student_features):
        """
        Accepts new student features and returns a human-readable performance prediction.
        
        Args:
            student_features (list or np.ndarray): Input features in the order:
                [study_hours, attendance, sleep_hours, assignments_completed (1/0), previous_gpa]
                
        Returns:
            str: The predicted performance category (e.g., 'Excellent', 'At Risk').
            
        Raises:
            Exception: If the prediction process fails.
        """
        try:
            # Prepare the input for the model (ensure it's a 2D array)
            features_array = np.array(student_features).reshape(1, -1)
            
            # Execute model prediction
            numeric_prediction = self.model.predict(features_array)
            
            # Map the numeric class back to a readable label
            readable_label = self.encoder.inverse_transform(numeric_prediction)[0]
            
            logger.info(f"Input features: {student_features} -> Prediction: {readable_label}")
            return readable_label

        except Exception as e:
            logger.error(f"Prediction failed for input {student_features}: {str(e)}")
            raise

    def predict_with_proba(self, student_features):
        """
        Performs prediction and returns the category along with the confidence score.
        
        Args:
            student_features (list): Input features.
            
        Returns:
            tuple: (predicted_label, confidence_score)
        """
        try:
            features_array = np.array(student_features).reshape(1, -1)
            
            # Get probabilities for all classes
            probabilities = self.model.predict_proba(features_array)[0]
            
            # Get the index of the highest probability
            max_prob_idx = np.argmax(probabilities)
            confidence = probabilities[max_prob_idx]
            
            # Get the label
            numeric_prediction = [self.model.classes_[max_prob_idx]]
            readable_label = self.encoder.inverse_transform(numeric_prediction)[0]
            
            return readable_label, confidence
            
        except Exception as e:
            logger.error(f"Detailed prediction failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Internal verification and demonstration
    print("\n--- Student Performance Prediction Test ---")
    try:
        predictor = PerformancePredictor()
        
        # Test Case 1: Likely 'Excellent'
        # [study_hours, attendance, sleep_hours, assignments_completed, previous_gpa]
        excellent_student = [9.0, 95, 8, 1, 3.9]
        res1 = predictor.predict(excellent_student)
        print(f"Input: {excellent_student} | Result: {res1}")
        
        # Test Case 2: Likely 'At Risk'
        at_risk_student = [1.5, 42, 4, 0, 1.4]
        res2 = predictor.predict(at_risk_student)
        print(f"Input: {at_risk_student} | Result: {res2}")
        
    except Exception as e:
        print(f"Prediction module test failed: {e}")
