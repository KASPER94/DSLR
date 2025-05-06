from utils.load_csv import load
import numpy as np
from utils.maths import _count, _mean, _std, _max, _min, _percentile

def start(file):
    datas = load(file)
    features = datas[0]
    dataset = datas[1:, :]

    numeric_cols = range(6, len(features))
    stat_names = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Range"]

    stats = {name: [] for name in stat_names}

    for col in numeric_cols:
        col_data = np.array([
            float(dataset[i, col])
            for i in range(len(dataset))
            if isinstance(dataset[i, col], float) and not np.isnan(dataset[i, col])
        ])

        stats["Count"].append(_count(col_data))
        stats["Mean"].append(_mean(col_data))
        stats["Std"].append(_std(col_data))
        stats["Min"].append(_min(col_data))
        stats["25%"].append(_percentile(col_data, 25))
        stats["50%"].append(_percentile(col_data, 50))
        stats["75%"].append(_percentile(col_data, 75))
        stats["Max"].append(_max(col_data))
        stats["Range"].append(_max(col_data) - _min(col_data))

    col_width = 22
    max_name_len = col_width - 4

    def format_header(name):
        return (name[:max_name_len] + '...') if len(name) > max_name_len else name.ljust(col_width)

    headers = [format_header(features[col]) for col in numeric_cols]

    print(f'{"Stat":15} ' + ''.join([f'| {h} ' for h in headers[:4]]) + '|')
    print('-' * (17 + len(headers[:4]) * (col_width + 3)))

    for stat in stat_names:
        row = ''.join([f'| {v:>{col_width }.2f} ' for v in stats[stat][:4]])
        print(f'{stat:15} {row}|')

    output_file = "describe_output.txt"
    with open(output_file, mode='w') as f:
        f.write(f'{"Stat":15} ' + ''.join([f'| {h} ' for h in headers]) + '|\n')
        f.write('-' * (17 + len(headers) * (col_width + 3)) + '\n')
        for stat in stat_names:
            row = ''.join([f'| {v:>{col_width}.2f} ' for v in stats[stat]])
            f.write(f'{stat:15} {row}|\n')

    print(f"\nStatistiques enregistrées dans `{output_file}`.")
