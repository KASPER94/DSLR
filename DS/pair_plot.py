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

    house_colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }

    palette = {
        house: color for house, color in house_colors.items()
        if house in df["Hogwarts House"].unique()
    }

    pair1 = sns.pairplot(df, vars=courses1, hue="Hogwarts House",
                         palette=palette, diag_kind="hist", plot_kws={"alpha": 0.6, "s": 20})
    pair1.fig.suptitle("Pair Plot - Bloc 1", y=1.02)
    plt.tight_layout()
    plt.show()

    pair2 = sns.pairplot(df, vars=courses2, hue="Hogwarts House",
                         palette=palette, diag_kind="hist", plot_kws={"alpha": 0.6, "s": 20})
    pair2.fig.suptitle("Pair Plot - Bloc 2", y=1.02)
    plt.tight_layout()
    plt.show()

def pair_plot_start(filepath):
    raw_data = load(filepath)
    headers = raw_data[0]
    dataset = raw_data[1:]

    df = pd.DataFrame(dataset, columns=headers)

    for col in df.columns[6:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=["Hogwarts House"] + list(df.columns[6:]))

    plot_pair(df)
