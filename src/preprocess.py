import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A robust preprocessing pipeline for the Student Performance Prediction System.
    Handles data loading, validation, imputation, encoding, and feature separation.
    """
    
    REQUIRED_COLUMNS = [
        'study_hours',
        'attendance',
        'sleep_hours',
        'assignments_completed',
        'previous_gpa',
        'performance'
    ]

    def __init__(self, data_path: str):
        """
        Initializes the preprocessor.
        
        Args:
            data_path (str): The path to the CSV file containing the dataset.
        """
        self.data_path = data_path
        self.df = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from the CSV file and validates required columns.
        
        Returns:
            pd.DataFrame: The loaded dataset.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the dataset is missing any required columns.
        """
        if not os.path.exists(self.data_path):
            error_msg = f"Data file not found at path: {self.data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
            
        # Validate that all required columns are present
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_cols:
            error_msg = f"Dataset is missing required columns: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Successfully loaded {len(self.df)} records.")
        return self.df

    def handle_missing_values(self):
        """
        Handles missing values safely using sklearn SimpleImputer.
        - Numerical features are imputed with their median.
        - Categorical features are imputed with the most frequent value.
        - Rows with missing target labels ('performance') are dropped.
        """
        logger.info("Handling missing values...")
        
        # Define feature groups
        num_cols = ['study_hours', 'attendance', 'sleep_hours', 'previous_gpa']
        cat_cols = ['assignments_completed']
        
        # Impute numerical features with the median value
        num_imputer = SimpleImputer(strategy='median')
        if not self.df[num_cols].empty:
            self.df[num_cols] = num_imputer.fit_transform(self.df[num_cols])
            
        # Impute categorical features with the most frequent value (mode)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        if not self.df[cat_cols].empty:
            self.df[cat_cols] = cat_imputer.fit_transform(self.df[cat_cols])
            
        # Drop rows where the target label 'performance' is missing
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['performance'])
        if len(self.df) < initial_len:
            dropped = initial_len - len(self.df)
            logger.warning(f"Dropped {dropped} rows due to missing 'performance' labels.")

    def encode_categorical_variables(self):
        """
        Encodes categorical variables into numerical format.
        - 'assignments_completed': Maps 'Yes' to 1 and 'No' to 0.
        """
        logger.info("Encoding categorical variables...")
        
        if 'assignments_completed' in self.df.columns:
            # Map text to binary values
            mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
            self.df['assignments_completed'] = self.df['assignments_completed'].map(mapping)
            
            # If there were any unexpected values that became NaN after mapping, fill with 0
            if self.df['assignments_completed'].isnull().any():
                logger.warning("Unexpected values found in 'assignments_completed'. Defaulting to 0.")
                self.df['assignments_completed'] = self.df['assignments_completed'].fillna(0)
                
            self.df['assignments_completed'] = self.df['assignments_completed'].astype(int)

    def separate_features_labels(self):
        """
        Separates the dataset into the feature matrix (X) and the target vector (y).
        Encodes the target labels into numeric classes.
        
        Returns:
            tuple: (X (np.ndarray), y (np.ndarray), feature_names (list))
        """
        logger.info("Separating features and labels...")
        
        # Extract features (X) and ensure ordering
        feature_columns = [col for col in self.REQUIRED_COLUMNS if col != 'performance']
        X_df = self.df[feature_columns]
        
        # Extract labels (y)
        y_series = self.df['performance']
        
        # Save feature names for future reference
        feature_names = X_df.columns.tolist()
        
        # Fit label encoder and transform labels to integer codes
        y = self.label_encoder.fit_transform(y_series)
        X = X_df.values
        
        return X, y, feature_names

    def process(self):
        """
        Executes the entire preprocessing pipeline.
        
        Returns:
            tuple: Processed feature matrix (X), encoded labels (y), and feature names.
        """
        try:
            self.load_data()
            self.handle_missing_values()
            self.encode_categorical_variables()
            X, y, feature_names = self.separate_features_labels()
            
            logger.info("Preprocessing completed successfully.")
            return X, y, feature_names
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage for testing the preprocessing module
    try:
        # Assuming script is run from the project root
        data_file = "data/student_data.csv"
        
        preprocessor = DataPreprocessor(data_path=data_file)
        X, y, features = preprocessor.process()
        
        print("\n--- Preprocessing Results ---")
        print(f"Feature Names: {features}")
        print(f"Feature Matrix (X) Shape: {X.shape}")
        print(f"Labels (y) Shape: {y.shape}")
        print(f"Label Mapping: {dict(zip(preprocessor.label_encoder.classes_, range(len(preprocessor.label_encoder.classes_))))}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
