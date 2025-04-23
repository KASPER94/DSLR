from load_csv import load
import numpy as np
from maths import count

def start(file):
    datas = load(file)
    features = datas[0]
    dataset = datas[1:, :]
    print(f'{"":15} |{features[6]:>12} |{features[7]:>12} |{features[8]:>12} | {features[9]:>12} |')
    all_col_data = np.array([])
    for i in range(0, len(dataset)):
        try:
            value = float(dataset[i, 6])
            if not np.isnan(value):
                all_col_data = np.append(all_col_data, value)
            # print(f'{"":15} | {count(data):>12.4f}')
        except:
            print("error")
    print(count(all_col_data))