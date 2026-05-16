import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A modular evaluation module for the Student Performance Prediction System.
    Calculates key classification metrics and provides formatted summaries.
    """
    
    def __init__(self, target_names: list = None):
        """
        Initializes the evaluator.
        
        Args:
            target_names (list, optional): Human-readable names for the target classes.
        """
        self.target_names = target_names

    def evaluate(self, y_true, y_pred):
        """
        Performs a full evaluation of the model predictions against ground truth.
        
        Args:
            y_true (np.ndarray): The ground truth labels.
            y_pred (np.ndarray): The predicted labels from the model.
            
        Returns:
            dict: A dictionary containing all calculated metrics.
            
        Raises:
            Exception: If evaluation fails due to inconsistent input shapes or other issues.
        """
        try:
            logger.info("Initiating model performance evaluation...")
            
            # Calculate standard metrics using 'weighted' average to account for class imbalance
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
                "confusion_matrix": confusion_matrix(y_true, y_pred)
            }

            self._display_readable_summary(metrics, y_true, y_pred)
            
            return metrics

        except Exception as e:
            logger.error(f"An error occurred during evaluation: {str(e)}")
            raise

    def analyze_feature_importance(self, model, feature_names):
        """
        Extracts and displays the importance of each feature from the model.
        
        Args:
            model: A trained model with a 'feature_importances_' attribute (e.g., RandomForest).
            feature_names (list): The list of feature labels corresponding to the model inputs.
            
        Returns:
            list: A sorted list of (feature, importance) tuples.
            
        Raises:
            AttributeError: If the model does not support feature importance.
            Exception: If the analysis fails.
        """
        try:
            logger.info("Analyzing feature importance...")
            
            if not hasattr(model, 'feature_importances_'):
                error_msg = "The model does not have the 'feature_importances_' attribute."
                logger.error(error_msg)
                raise AttributeError(error_msg)
                
            importances = model.feature_importances_
            
            # Pair feature names with their importance scores and sort descending
            feature_importance = sorted(
                zip(feature_names, importances), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            self._display_importance_summary(feature_importance)
            
            return feature_importance

        except Exception as e:
            logger.error(f"Feature importance analysis failed: {str(e)}")
            raise

    def _display_importance_summary(self, feature_importance):
        """
        Prints a clean, sorted report showing which factors most influence predictions.
        """
        print("\n" + "-"*50)
        print(" FEATURE IMPORTANCE ANALYSIS ".center(50))
        print("-" * 50)
        
        print(f"{'Feature Name':<30} | {'Importance':<10}")
        print("-" * 45)
        
        for feature, score in feature_importance:
            print(f"{feature.replace('_', ' ').title():<30} | {score:.4f}")
            
        print("\n[INSIGHT] Factors with higher scores have the most significant")
        print("influence on predicting a student's performance category.")
        print("-" * 50 + "\n")

    def _display_readable_summary(self, metrics, y_true, y_pred):
        """
        Prints a presentation-friendly summary of the model's performance to the terminal.
        """
        print("\n" + "="*70)
        print(" STUDENT PERFORMANCE MODEL EVALUATION REPORT ".center(70))
        print("="*70)
        
        print(f"{'Core Metrics':<30} | {'Score':<10}")
        print("-" * 45)
        print(f"{'Overall Accuracy':<30} | {metrics['accuracy']:.4f}")
        print(f"{'Weighted Precision':<30} | {metrics['precision']:.4f}")
        print(f"{'Weighted Recall':<30} | {metrics['recall']:.4f}")
        print(f"{'Weighted F1-Score':<30} | {metrics['f1_score']:.4f}")
        
        print("\n" + "--- Detailed Classification Metrics ---")
        print(classification_report(y_true, y_pred, target_names=self.target_names, zero_division=0))
        
        print("\n" + "--- Confusion Matrix (Row: Actual, Column: Predicted) ---")
        self._print_formatted_cm(metrics['confusion_matrix'])
            
        print("="*70 + "\n")

    def _print_formatted_cm(self, cm):
        """
        Helper method to print a pretty-formatted confusion matrix.
        """
        if self.target_names is not None:
            # Create a header with class names
            header = " " * 15 + " ".join([f"{name[:10]:>12}" for name in self.target_names])
            print(header)
            print(" " * 15 + "-" * (13 * len(self.target_names)))
            
            for i, row in enumerate(cm):
                row_label = f"{self.target_names[i][:12]:>12} | "
                row_vals = " ".join([f"{val:>12}" for val in row])
                print(row_label + row_vals)
        else:
            print(cm)

if __name__ == "__main__":
    # Independent verification with dummy data
    print("Running evaluate.py verification...")
    mock_classes = ['Excellent', 'Good', 'Average', 'At Risk']
    mock_y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    mock_y_pred = np.array([0, 1, 1, 3, 0, 1, 2, 3, 0, 2])
    
    evaluator = ModelEvaluator(target_names=mock_classes)
    evaluator.evaluate(mock_y_true, mock_y_pred)
