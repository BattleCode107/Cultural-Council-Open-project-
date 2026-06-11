"""
data_processing.py

This module handles the extraction, filtering, and preprocessing of the Netflix Prize Dataset.
Due to memory constraints (16GB RAM limit), it aggressive filters out movies with < 500 ratings
and users with < 50 ratings to create a manageable subset.
"""
import os
import time
from collections import defaultdict

def preprocess_dataset(data_dir, output_file):
    """
    Reads combined_data_1.txt to combined_data_4.txt, computes frequencies,
    filters the dataset, and writes to a structured CSV.
    """
    files = [f"combined_data_{i}.txt" for i in range(1, 5)]
    user_counts = defaultdict(int)
    movie_counts = defaultdict(int)
    
    # Pass 1: Count Frequencies
    print("Pass 1: Counting user and movie frequencies...")
    for file in files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath): continue
        current_movie = None
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.endswith(':'):
                    current_movie = int(line[:-1])
                else:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        user_id = int(parts[0])
                        user_counts[user_id] += 1
                        movie_counts[current_movie] += 1
                        
    # Identify valid users and movies
    valid_users = {uid for uid, count in user_counts.items() if count > 50}
    valid_movies = {mid for mid, count in movie_counts.items() if count > 500}
    
    # Pass 2: Write Filtered Data
    print("\nPass 2: Filtering and writing to CSV...")
    with open(output_file, 'w') as out_f:
        out_f.write("user_id,movie_id,rating\n")
        for file in files:
            filepath = os.path.join(data_dir, file)
            if not os.path.exists(filepath): continue
            current_movie = None
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if line.endswith(':'):
                        current_movie = int(line[:-1])
                    else:
                        if current_movie in valid_movies:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                user_id = int(parts[0])
                                rating = int(parts[1])
                                if user_id in valid_users:
                                    out_f.write(f"{user_id},{current_movie},{rating}\n")

if __name__ == '__main__':
    data_dir = r"."
    output_file = os.path.join(data_dir, "filtered_dataset.csv")
    preprocess_dataset(data_dir, output_file)
