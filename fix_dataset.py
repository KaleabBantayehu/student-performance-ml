import pandas as pd
import numpy as np

def generate_realistic_data(num_samples=1000):
    np.random.seed(42)
    
    # Generate somewhat independent features
    study_hours = np.random.normal(loc=12, scale=5, size=num_samples)
    study_hours = np.clip(study_hours, 2, 25)
    
    attendance = np.random.normal(loc=85, scale=10, size=num_samples)
    attendance = np.clip(attendance, 50, 100)
    
    sleep_hours = np.random.normal(loc=7, scale=1.5, size=num_samples)
    sleep_hours = np.clip(sleep_hours, 4, 10)
    
    previous_gpa = np.random.normal(loc=3.0, scale=0.6, size=num_samples)
    previous_gpa = np.clip(previous_gpa, 1.0, 4.0)
    
    # Assignments completed: introduce independence but slight correlation
    # For example, if you study more, you are more likely to complete assignments
    prob_assignment = 1 / (1 + np.exp(- (0.3 * (study_hours - 12) + 0.1 * (attendance - 80))))
    # Clip prob to [0.1, 0.9] to ensure variance
    prob_assignment = np.clip(prob_assignment, 0.2, 0.95)
    
    assignments_completed_num = np.random.binomial(1, prob_assignment)
    assignments_completed = ['Yes' if x == 1 else 'No' for x in assignments_completed_num]
    
    # Calculate an underlying score to determine performance
    # Weighting: 
    # Study: 25%, Attendance: 30%, Assignments: 20%, GPA: 15%, Sleep: 10%
    
    norm_sh = (study_hours - 2) / 23
    norm_att = (attendance - 50) / 50
    norm_gpa = (previous_gpa - 1) / 3
    norm_slp = (sleep_hours - 4) / 6
    
    score = (0.25 * norm_sh + 
             0.30 * norm_att + 
             0.20 * assignments_completed_num + 
             0.15 * norm_gpa + 
             0.10 * norm_slp)
             
    # Add some random noise to the score to prevent deterministic perfect splits
    score += np.random.normal(0, 0.05, size=num_samples)
    
    # Map score to performance
    performance = []
    for s in score:
        if s > 0.75:
            performance.append("Excellent")
        elif s > 0.60:
            performance.append("Good")
        elif s > 0.45:
            performance.append("Average")
        else:
            performance.append("At Risk")
            
    df = pd.DataFrame({
        'study_hours': np.round(study_hours, 1),
        'attendance': np.round(attendance, 0),
        'sleep_hours': np.round(sleep_hours, 1),
        'assignments_completed': assignments_completed,
        'previous_gpa': np.round(previous_gpa, 2),
        'performance': performance
    })
    
    df.to_csv("data/student_data.csv", index=False)
    print("Dataset successfully rebalanced and regenerated.")

if __name__ == "__main__":
    generate_realistic_data()
