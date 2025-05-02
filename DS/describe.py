from utils.load_csv import load
import numpy as np
from utils.maths import _count, _mean, _std, _max, _min, _percentile

def start(file):
    datas = load(file)
    features = datas[0]
    dataset = datas[1:, :]
    print(f'{"":15} |{features[6]:>12} |{features[7]:>12} |{features[8]:>12} | {features[9]:>12} |')
    count = []
    mean = []
    Std = []
    Min = []
    Max = []
    twentyFive = []
    fifty = []
    sevFiv = []
    for col in range(6, 10):
        all_col_data = np.array([])
        
        for i in range(0, len(dataset)):
            try:
                value = float(dataset[i, col])
                if not np.isnan(value):
                    all_col_data = np.append(all_col_data, value)
            except:
                pass 
        
        count.append(_count(all_col_data))
        mean.append(_mean(all_col_data))
        Std.append(_std(all_col_data))
        Min.append(_min(all_col_data))
        Max.append(_max(all_col_data))
        twentyFive.append(_percentile(all_col_data, 25))
        fifty.append(_percentile(all_col_data, 50))
        sevFiv.append(_percentile(all_col_data, 75))
    print(f'{"Count":15} | {count[0]:>12.2f}| {count[1]:>12.2f}| {count[2]:>12.2f}| {count[3]:>12.2f}                  |')
    print(f'{"Mean":15} | {mean[0]:>12.2f}| {mean[1]:>12.2f}| {mean[2]:>12.2f}| {mean[3]:>12.2f}                  |')
    print(f'{"Std":15} | {Std[0]:>12.2f}| {Std[1]:>12.2f}| {Std[2]:>12.2f}| {Std[3]:>12.2f}                  |')
    print(f'{"Min":15} | {Min[0]:>12.2f}| {Min[1]:>12.2f}| {Min[2]:>12.2f}| {Min[3]:>12.2f}                  |')
    print(f'{"25%":15} | {twentyFive[0]:>12.2f}| {twentyFive[1]:>12.2f}| {twentyFive[2]:>12.2f}| {twentyFive[3]:>12.2f}                  |')
    print(f'{"50%":15} | {fifty[0]:>12.2f}| {fifty[1]:>12.2f}| {fifty[2]:>12.2f}| {fifty[3]:>12.2f}                  |')
    print(f'{"75%":15} | {sevFiv[0]:>12.2f}| {sevFiv[1]:>12.2f}| {sevFiv[2]:>12.2f}| {sevFiv[3]:>12.2f}                  |')
    print(f'{"Max":15} | {Max[0]:>12.2f}| {Max[1]:>12.2f}| {Max[2]:>12.2f}| {Max[3]:>12.2f}                  |')
