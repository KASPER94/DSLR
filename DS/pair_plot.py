import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.load_csv import load

def plot_pair(df):
    courses = df.columns[6:]
    mid_index = len(courses) // 2
    courses1 = courses[:mid_index]
    courses2 = courses[mid_index:]

    # sns.set(style="ticks")

    # Bloc 1
    pair1 = sns.pairplot(df, vars=courses1, hue="Hogwarts House", palette="bright", diag_kind="hist")
    pair1.fig.suptitle("Pair Plot - Bloc 1", y=1.02)
    plt.tight_layout()
    plt.show()
    # pair1.savefig("pair_plot_1.png")  # décommente pour sauvegarder

    # Bloc 2
    pair2 = sns.pairplot(df, vars=courses2, hue="Hogwarts House", palette="bright", diag_kind="hist")
    pair2.fig.suptitle("Pair Plot - Bloc 2", y=1.02)
    plt.tight_layout()
    plt.show()
    # pair2.savefig("pair_plot_2.png")  # décommente pour sauvegarder

def pair_plot_start(filepath):
    raw_data = load(filepath)
    headers = raw_data[0]
    dataset = raw_data[1:]

    # Créer un DataFrame propre
    df = pd.DataFrame(dataset, columns=headers)

    # Convertir les colonnes numériques
    for col in df.columns[6:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    plot_pair(df)
