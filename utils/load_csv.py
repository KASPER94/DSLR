import sys
import csv
import numpy as np

def error_exit(str):
    """
    Print error and exit
    """
    print(str)
    sys.exit(0)

def checkfile(file):
    """
    Chcking if file is correct
    """
    try:
        with open(file, 'r') as f:
            fl = f.readlines()
            for l in fl[1:]:
                L = l[:-1].split(',')
                if not L[0].isalpha() and not L[0].isnumeric():
                    return 0
            return 1
    except:
        error_exit("No data")

def load(file):
    checkfile(file)
    dataset = list()
    with open(file) as csvDatafile:
        csvReader = csv.reader(csvDatafile)
        try:
            for r in csvReader:
                row = list()
                for value in r:
                    try:
                        value = float(value)
                    except:
                        if not value:
                            value = np.nan
                    row.append(value)
                dataset.append(row)
        except csv.Error as e:
            print(f"error file : {file}, line {csvReader.line_num}: {e}")
    return np.array(dataset, dtype=object)
