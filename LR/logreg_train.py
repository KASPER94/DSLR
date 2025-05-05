from utils.load_csv import load, checkfile
import pandas as pd
import numpy as np

class Train:
    def __init__(self):
        self.dataset = list()
        self.X = None
        self.y = None
        self.classes = []
        self.cost = None
        self.theta = None

    def load_and_prepare_data(self, file):
        raw = load(file)
        headers = raw[0]
        dataset = raw[1:]

        df = pd.DataFrame(dataset, columns=headers)

        for col in df.columns[6:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=["Hogwarts House"] + list(df.columns[6:]))

        X = df.iloc[:, 6:].values
        y = df["Hogwarts House"].values

        self.X = X
        self.y = y
        self.classes = np.unique(y)

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, theta):
        m = len(y)
        h = Train.sigmoid(np.dot(X, theta))
        epsilon = 1e-15
        self.cost = (-1 / m) * np.sum(
            y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)
        )

    def gradient_descent(self, X, y, theta, alpha, ite):
        m = len(y)
        for i in range(ite):
            h = Train.sigmoid(np.dot(X, theta))
            gradient = (1 / m) * np.dot(X.T, (h - y))
            theta -= alpha * gradient
        self.theta = theta

    def train_one_vs_ALL(self, X, y, classes, alpha, ite):
        m, n = X.shape
        thetas = {}

        # Ajouter une colonne de biais (x0 = 1)
        X = np.concatenate([np.ones((m, 1)), X], axis=1)

        for class_name in classes:
            y_binary = np.array([1 if label == class_name else 0 for label in y])
            theta = np.zeros(n + 1)

            self.gradient_descent(X, y_binary, theta, alpha, ite)
            thetas[class_name] = self.theta

        return thetas

    @staticmethod
    def save_weights(thetas, filename):
        houses = list(thetas.keys())
        theta_matrix = np.array([thetas[house] for house in houses])

        n = theta_matrix.shape[1]
        header = ['House'] + [f'theta_{i}' for i in range(n)]

        df = pd.DataFrame(theta_matrix, index=houses, columns=header[1:])
        df.insert(0, "House", houses)

        df.to_csv(filename, index=False)


def train(file):
    model = Train()
    model.load_and_prepare_data(file)
    thetas = model.train_one_vs_ALL(model.X, model.y, model.classes, alpha=0.1, ite=1000)
    model.save_weights(thetas, "weights.csv")

