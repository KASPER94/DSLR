import matplotlib.pyplot as plt
import pandas as pd
from utils.load_csv import load

def scatter_plot_start(filepath):
    raw_data = load(filepath)
    headers = raw_data[0]
    dataset = raw_data[1:]

    df = pd.DataFrame(dataset, columns=headers)

    for col in df.columns[6:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    x_feature = "Astronomy"
    y_feature = "Defense Against the Dark Arts"

    df = df.dropna(subset=["Hogwarts House", x_feature, y_feature])

    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    house_colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }

    plt.figure(figsize=(8, 6))
    for house in houses:
        house_data = df[df["Hogwarts House"] == house]
        plt.scatter(house_data[x_feature], house_data[y_feature],
                    label=house, alpha=0.7, color=house_colors[house])

    plt.title(f"Scatter Plot: {x_feature} vs {y_feature}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
