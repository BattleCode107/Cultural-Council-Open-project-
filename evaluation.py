"""
evaluation.py

This module contains the custom evaluation logic for the models, strictly implementing
RMSE and MAP@10 (with a relevance threshold of >= 3.5 stars) as defined by the rubric.
"""
from surprise import accuracy
from collections import defaultdict

def get_top_n_and_map_at_10(predictions, n=10, threshold=3.5):
    """
    Calculates the MAP@10 metric and returns the top-N recommendations.
    """
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est, true_r))

    # Sort predictions for each user
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    # Calculate MAP@10
    user_aps = []
    for uid, user_ratings in top_n.items():
        hits = 0
        sum_precisions = 0
        for i, (iid, est, true_r) in enumerate(user_ratings):
            if true_r >= threshold:
                hits += 1
                sum_precisions += hits / (i + 1.0)
                
        if len(user_ratings) > 0:
            ap = sum_precisions / min(n, len(user_ratings))
            user_aps.append(ap)

    map_at_10 = sum(user_aps) / len(user_aps) if user_aps else 0
    return top_n, map_at_10

def evaluate_model(algo, testset, model_name="Model"):
    """
    Generates predictions and prints RMSE and MAP@10.
    """
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    _, map10 = get_top_n_and_map_at_10(predictions, n=10, threshold=3.5)
    
    print(f"{model_name} Results:")
    print(f"  -> RMSE:   {rmse:.4f}")
    print(f"  -> MAP@10: {map10:.4f}")
    
    return predictions
