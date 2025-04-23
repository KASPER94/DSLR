from load_csv import load
import numpy as np

def start(file):
    data = load(file)
    print(data.shape)