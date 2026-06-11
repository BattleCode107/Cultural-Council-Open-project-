# Netflix Prize Recommendation System

A machine learning project focused on personalized content discovery using the Netflix Prize Dataset.

This repository implements and evaluates multiple recommendation system approaches, including **Item-Based Collaborative Filtering (KNN)** and **Matrix Factorization (SVD)**, to predict user preferences and generate personalized movie recommendations.

The project was developed under standard hardware constraints (16GB RAM) using a filtered subset of the Netflix Prize Dataset while preserving the characteristics of large-scale recommendation problems such as sparsity, popularity bias, and long-tail content distribution.

---

## Project Objectives

The recommendation engine is designed to:

* Learn user preferences from historical ratings
* Predict ratings for unseen movies
* Generate personalized Top-K recommendations
* Compare traditional and latent-factor recommendation approaches
* Evaluate recommendation quality using both prediction and ranking metrics

---

## Dataset

This project uses the **Netflix Prize Dataset**, one of the most influential benchmark datasets in recommender systems research.

### Dataset Statistics

| Metric       | Value       |
| ------------ | ----------- |
| Ratings      | 100,480,507 |
| Users        | 480,189     |
| Movies       | 17,770      |
| Rating Scale | 1–5 Stars   |

### Key Challenges

* Extreme sparsity (~98.8%)
* Long-tail content distribution
* Popularity bias
* Cold-start scenarios
* Large-scale user-item interactions

To enable efficient experimentation, the dataset is filtered and sampled before training.

---

## Recommendation Models Implemented

### 1. Item-Based Collaborative Filtering (KNN)

A memory-based recommendation approach that:

* Computes similarity between movies
* Uses Pearson Baseline similarity
* Recommends items based on neighboring movies

**Advantages**

* Intuitive and explainable
* Simple implementation

**Limitations**

* Struggles with sparse data
* High memory requirements
* Poor scalability for large catalogs

---

### 2. Matrix Factorization (SVD)

A latent-factor recommendation approach that:

* Decomposes the user-item matrix into latent features
* Learns hidden user preferences
* Learns hidden movie characteristics

Configuration:

* 50 latent factors
* Stochastic Gradient Descent optimization
* Surprise library implementation

**Advantages**

* Handles sparse data effectively
* Scales efficiently
* Fast inference
* Better recommendation quality

---

## Repository Structure

```text
archive/
├── README.md
├── movie_titles.csv
├── combined_data_1.txt
├── combined_data_2.txt
├── combined_data_3.txt
├── combined_data_4.txt
├── filtered_dataset.csv          # Generated after preprocessing

└── src/
    ├── data_processing.py
    ├── model_training.py
    ├── evaluation.py
    └── recommendation.py
```

### Module Description

| File               | Purpose                                            |
| ------------------ | -------------------------------------------------- |
| data_processing.py | Dataset parsing, filtering, and preprocessing      |
| model_training.py  | Model training pipeline for SVD and KNN            |
| evaluation.py      | RMSE and MAP@10 evaluation                         |
| recommendation.py  | Recommendation generation and qualitative analysis |

---

## Installation

### Prerequisites

* Python 3.10+
* 16GB RAM recommended

### Install Dependencies

```bash
pip install scikit-surprise pandas numpy matplotlib seaborn
```

---

## Reproducing Results

### Step 1 — Data Preprocessing

Parse the raw Netflix dataset and generate the filtered dataset.

```bash
python src/data_processing.py
```

This script:

* Loads raw Netflix rating files
* Filters users with more than 50 ratings
* Filters movies with more than 500 ratings
* Generates:

```text
filtered_dataset.csv
```

---

### Step 2 — Train and Evaluate Models

Run the training pipeline.

```bash
python src/model_training.py
```

The script:

* Samples 1,000,000 ratings
* Creates an 80/20 train-test split
* Trains:

  * Item-Based KNN
  * SVD Matrix Factorization
* Evaluates:

  * RMSE
  * MAP@10

---

### Step 3 — Generate Recommendation Examples

Recommendation examples are automatically generated at the end of training.

Alternatively:

```bash
python src/recommendation.py
```

This module displays:

* Top recommendations
* Success cases
* Failure cases
* Recommendation insights

---

## Evaluation Methodology

### Train-Test Split

* 80% Training Data
* 20% Testing Data
* Fixed random seed (`random_state=42`) for reproducibility

### Metrics

#### RMSE (Root Mean Squared Error)

Measures rating prediction accuracy.

Lower values indicate better performance.

#### MAP@10 (Mean Average Precision @ 10)

Measures recommendation ranking quality.

A movie is considered relevant when:

```text
Actual Rating ≥ 3.5
```

Higher values indicate better recommendation quality.

---

## Experimental Results

| Model                          | RMSE       | MAP@10     |
| ------------------------------ | ---------- | ---------- |
| **SVD (Matrix Factorization)** | **0.9661** | **0.5953** |
| Item-Based KNN                 | 1.1318     | 0.5895     |

### Key Findings

* SVD achieved the lowest prediction error.
* SVD delivered the highest recommendation ranking quality.
* Latent-factor models handled sparsity significantly better than memory-based approaches.
* SVD proved more scalable and production-ready.

---

## Recommendation Insights

### Success Cases

Examples where the model accurately captured user preferences:

* North by Northwest
* Aqua Teen Hunger Force: Vol. 1

### Failure Cases

Examples illustrating cold-start and sparsity challenges:

* Lady Chatterley
* I Love Lucy: Season 2

These cases highlight the limitations of collaborative filtering under sparse interaction conditions.

---

## Future Improvements

Potential enhancements include:

* Hybrid recommendation systems
* Content-based filtering
* Distributed training with Spark or Dask
* Hyperparameter optimization
* Ensemble recommendation models
* Real-time recommendation serving

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Surprise
* Matplotlib
* Collaborative Filtering
* Matrix Factorization (SVD)

---

## Reproducibility

All experiments were conducted using a fixed random seed and a documented preprocessing pipeline to ensure reproducibility of reported results.

---

## License

This project is intended for educational and research purposes using the Netflix Prize Dataset.

Dataset used can be found in the following link along with the filtered dataset obtained after data processing.
https://drive.google.com/drive/folders/1QI6Aqv2zfBao6qEBPF7s1778VGNKlBlO?usp=sharing
