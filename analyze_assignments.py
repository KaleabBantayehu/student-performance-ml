import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def run_diagnostics(data_path="data/student_data.csv"):
    print("=== ASSIGNMENTS COMPLETED FEATURE DIAGNOSTIC REPORT ===")
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\n1. Data Overview:")
    print(df.head())
    
    print("\n2. Missing Values:")
    print(df.isnull().sum())

    if 'assignments_completed' in df.columns:
        mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
        df['assignments_completed_num'] = df['assignments_completed'].map(mapping).fillna(0)
            
        print("Distribution of assignments_completed:")
        print(df['assignments_completed_num'].value_counts(normalize=True))
        print("Variance of assignments_completed:", df['assignments_completed_num'].var())
    else:
        print("Column 'assignments_completed' not found!")
        return

    print("\n4. Feature Correlations:")
    # Only correlate numeric
    num_cols = ['study_hours', 'attendance', 'sleep_hours', 'previous_gpa', 'assignments_completed_num']
    if all(col in df.columns for col in num_cols):
        corr_matrix = df[num_cols].corr()
        print(corr_matrix['assignments_completed_num'].sort_values(ascending=False))
        
        # Check if it perfectly correlates with another feature
        high_corr = corr_matrix['assignments_completed_num'][abs(corr_matrix['assignments_completed_num']) > 0.8]
        if len(high_corr) > 1:
            print("\nWARNING: High multicollinearity detected!")
            print(high_corr)
    
    print("\n5. Target Variable Relationship:")
    if 'performance' in df.columns:
        print("Performance distribution given assignment completion:")
        print(pd.crosstab(df['performance'], df['assignments_completed_num'], normalize='columns'))
        
    print("\n6. Model Retrain Test (to check importance):")
    # Drop rows with NA
    df = df.dropna()
    X = df[num_cols]
    y = LabelEncoder().fit_transform(df['performance'])
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print("\nFeature Importances:")
    print(importances.sort_values(ascending=False))

if __name__ == "__main__":
    run_diagnostics()
