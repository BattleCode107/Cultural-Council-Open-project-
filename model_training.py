"""
model_training.py

This module defines the training pipeline for both the Matrix Factorization (SVD)
and Collaborative Filtering (KNN) models using the scikit-surprise library.
"""
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from evaluation import evaluate_model

def train_models(data_file):
    """
    Loads a subset of the filtered data, splits it, and trains both models.
    """
    print("Loading data subset (1M rows for strict 16GB RAM compliance)...")
    df = pd.read_csv(data_file, nrows=1000000)
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    
    print("Splitting into train and test sets (80/20)...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # 1. Train Matrix Factorization (SVD)
    print("\n--- Training Matrix Factorization (SVD) ---")
    algo_svd = SVD(n_factors=50, random_state=42)
    algo_svd.fit(trainset)
    evaluate_model(algo_svd, testset, "SVD")
    
    # 2. Train Collaborative Filtering (Item-Based KNN)
    print("\n--- Training Collaborative Filtering (Item-Based KNN) ---")
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    algo_knn = KNNBasic(sim_options=sim_options)
    algo_knn.fit(trainset)
    evaluate_model(algo_knn, testset, "Item-Based CF")
    
    return algo_svd, algo_knn

if __name__ == '__main__':
    train_models(r"../filtered_dataset.csv")
