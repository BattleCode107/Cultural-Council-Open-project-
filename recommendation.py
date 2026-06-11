"""
recommendation.py

This module isolates specific success and failure examples from the trained models
to qualitatively demonstrate recommendation quality.
"""
import pandas as pd
import os

def print_recommendation_cases(predictions, movies_file):
    """
    Identifies and prints 2 highly accurate (Success) and 2 highly inaccurate (Failure) predictions.
    """
    movies_df = pd.read_csv(movies_file, encoding='ISO-8859-1', header=None, 
                            names=['MovieID', 'Year', 'Name'], on_bad_lines='skip')
    movies_df.set_index('MovieID', inplace=True)
    
    successes, failures = [], []
    for uid, iid, true_r, est, _ in predictions:
        error = abs(true_r - est)
        if true_r >= 4.0 and est >= 4.0 and error < 0.2:
            successes.append((uid, iid, true_r, est, error))
        elif error > 3.0:
            failures.append((uid, iid, true_r, est, error))
            
    successes.sort(key=lambda x: x[4])
    failures.sort(key=lambda x: x[4], reverse=True)
    
    print("\n--- Recommendation Deep Dive ---")
    print("2 Success Cases (Accurate Predictions):")
    for uid, iid, true_r, est, _ in successes[:2]:
        name = movies_df.loc[iid, 'Name'] if iid in movies_df.index else "Unknown"
        print(f"  User {uid} -> {name} | Actual: {true_r} | Predicted: {est:.2f}")
        
    print("\n2 Failure Cases (Inaccurate Predictions):")
    for uid, iid, true_r, est, _ in failures[:2]:
        name = movies_df.loc[iid, 'Name'] if iid in movies_df.index else "Unknown"
        print(f"  User {uid} -> {name} | Actual: {true_r} | Predicted: {est:.2f}")
