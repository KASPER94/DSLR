import numpy as np
import pandas as pd
import json

class Predict:
    def __init__(self, weights_file):
        self.weights = pd.read_csv(weights_file)
        self.classes = self.weights["House"].values
        self.thetas = self.weights.drop(columns=["House"]).values

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def preprocess(self, df, selected_features, feature_means):
        for col in selected_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(feature_means[col])
        X = df[selected_features].values
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # normalisation z-score
        X = np.insert(X, 0, 1, axis=1)  # ajout du biais
        return df.index.values, X

    def predict(self, X):
        probs = self.sigmoid(np.dot(X, self.thetas.T))  # shape (n, num_classes)
        preds = np.argmax(probs, axis=1)
        return [self.classes[i] for i in preds]


def predict(args):
    test_file = args[0]
    df = pd.read_csv(test_file)

    # mêmes features que dans train()
    selected_features = ['Astronomy', 'Ancient Runes', 'Transfiguration',
                         'Charms', 'Herbology', 'Defense Against the Dark Arts']

    # calcul des moyennes globales sur les colonnes (pas de maison connue dans test)
    feature_means = {}
    for col in selected_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        feature_means[col] = df[col].mean()

    model = Predict("weights.csv")
    index, X = model.preprocess(df, selected_features, feature_means)
    predictions = model.predict(X)

    result = pd.DataFrame({
        "Index": index,
        "Hogwarts House": predictions
    })
    result.to_csv("houses.csv", index=False)
    print("Prédictions enregistrées dans houses.csv")
