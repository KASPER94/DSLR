import matplotlib.pyplot as plt
from utils.load_csv import load

def hist_for_each(file):
    data = load(file)
    headers = data[0]
    dataset = data[1:, :]

    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    house_colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }
    for i in range(6, data.shape[1]):
        course_name = headers[i]
        plt.figure()
        for house in houses:
            scores = []
            for row in dataset:
                if row[1] == house and isinstance(row[i], float):
                    scores.append(row[i])
            plt.hist(scores, bins=10, alpha=0.5, label=house, color=house_colors[house])
        plt.title(f"Histogram of {course_name}")
        plt.xlabel("Scores")
        plt.ylabel("Number of Students")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def hist_start(file):
    data = load(file)
    headers = data[0]
    dataset = data[1:, :]

    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    house_colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))
    axes = axes.flatten()
    for i in range(6, data.shape[1]):
        ax = axes[i - 6]
        course_name = headers[i]
        for house in houses:
            scores = []
            for row in dataset:
                if row[1] == house and isinstance(row[i], float):
                    scores.append(row[i])
            ax.hist(scores, bins=10, alpha=0.5, label=house, color=house_colors[house])
        ax.set_title(f"Histogram of {course_name}")
        ax.set_xlabel("Scores")
        ax.set_ylabel("Number of Students")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()