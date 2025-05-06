import random
import numpy as np
import pandas as pd
import os
import itertools

def split_train_test(X, y, test_ratio=0.2, seed=42):
    np.random.seed(seed * os.getpid())
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return (X[train_indices], X[test_indices], y[train_indices], y[test_indices])

def accuracy_score(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def evaluate(self, thetas):
    X_test_bias = np.concatenate([np.ones((self.X_test.shape[0], 1)), self.X_test], axis=1)

    predictions = []
    for x in X_test_bias:
        scores = {house: self.sigmoid(np.dot(x, thetas[house])) for house in self.classes}
        predicted = max(scores.items(), key=lambda item: item[1])[0]
        predictions.append(predicted)

    acc = accuracy_score(self.y_test, predictions)
    print(f" Accuracy on test set (20%): {acc:.2%}")
    return acc

def score_feature_pair(df, feature_x, feature_y):
    houses = df["Hogwarts House"].unique()
    centroids = {}
    intra_variances = []

    for house in houses:
        subset = df[df["Hogwarts House"] == house][[feature_x, feature_y]].dropna()
        if len(subset) == 0:
            return 0
        centroid = subset.mean().values
        centroids[house] = centroid

        distances = np.linalg.norm(subset.values - centroid, axis=1)
        intra_variances.append(np.mean(distances))

    avg_intra_variance = np.mean(intra_variances)

    inter_class_distance = sum(
        np.linalg.norm(centroids[h1] - centroids[h2])
        for h1, h2 in itertools.combinations(houses, 2)
    )

    score = inter_class_distance / (1 + avg_intra_variance)

    return score

def rank_feature_pairs(df):
    features = df.columns[6:]
    ranked = []

    for f1, f2 in itertools.combinations(features, 2):
        score = score_feature_pair(df, f1, f2)
        ranked.append(((f1, f2), score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

def save_means(train_file, output_file="feature_means.json"):
    df = pd.read_csv(train_file)

    selected_features = ['Astronomy', 'Ancient Runes', 'Transfiguration',
                         'Charms', 'Herbology', 'Defense Against the Dark Arts']

    means = {}
    for col in selected_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        means[col] = df[col].mean()

    with open(output_file, "w") as f:
        json.dump(means, f)

    print(f"Feature means saved to {output_file}")