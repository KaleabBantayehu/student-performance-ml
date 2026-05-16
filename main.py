import logging
import sys
from src.preprocess import DataPreprocessor
from src.train_model import StudentModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import PerformancePredictor

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_ml_workflow():
    """
    Coordinates the end-to-end Machine Learning workflow for the 
    Student Performance Prediction System.
    
    Workflow Sequence:
    1. Preprocessing: Load, validate, and encode the raw dataset.
    2. Training: Split data and train the RandomForest model.
    3. Evaluation: Generate performance metrics and confusion matrix.
    4. Inference: Run a sample prediction to verify end-to-end functionality.
    """
    
    print("\n" + "="*75)
    print(" STUDENT PERFORMANCE PREDICTION SYSTEM: END-TO-END WORKFLOW ".center(75))
    print("="*75)

    # 1. Configuration
    data_path = "data/student_data.csv"
    
    try:
        # --- PHASE 1: PREPROCESSING ---
        print("\n[PHASE 1] DATA PREPROCESSING")
        print("-" * 30)
        preprocessor = DataPreprocessor(data_path=data_path)
        X, y, feature_names = preprocessor.process()
        print(f"[OK] Data processed successfully. Total Samples: {len(X)}")

        # --- PHASE 2: MODEL TRAINING & PERSISTENCE ---
        print("\n[PHASE 2] MODEL TRAINING & PERSISTENCE")
        print("-" * 30)
        trainer = StudentModelTrainer()
        # The trainer handles splitting, training, and automatic artifact saving
        model, X_test, y_test = trainer.train(X, y, encoder=preprocessor.label_encoder)
        print(f"[OK] Model and Encoder saved to 'models/' directory.")

        # --- PHASE 3: EVALUATION ---
        # The evaluator provides detailed metrics, a confusion matrix, and feature importance analysis
        print("\n[PHASE 3] PERFORMANCE EVALUATION")
        print("-" * 30)
        evaluator = ModelEvaluator(target_names=preprocessor.label_encoder.classes_)
        y_pred = model.predict(X_test)
        evaluator.evaluate(y_test, y_pred)
        
        # Analyze which features most influence the predictions
        evaluator.analyze_feature_importance(model, feature_names)

        # --- PHASE 4: SAMPLE INFERENCE (PREDICTION) ---
        print("\n[PHASE 4] SAMPLE INFERENCE TEST")
        print("-" * 30)
        predictor = PerformancePredictor()
        
        # Sample Test: High engagement student
        # Order: [study_hours, attendance, sleep_hours, assignments_completed, previous_gpa]
        sample_input = [8.5, 92, 7.5, 1, 3.7]
        result = predictor.predict(sample_input)
        
        print(f"\nTEST DATA  : {sample_input}")
        print(f"PREDICTION : {result}")
        
        print("\n" + "="*75)
        print(" WORKFLOW EXECUTION COMPLETED SUCCESSFULLY ".center(75))
        print("="*75 + "\n")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed: {str(e)}")
        logger.error(f"Execution halted during workflow: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_ml_workflow()