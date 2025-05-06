# DSLR
commands: 

Describe dataset: 
    python main.py -d datasets/dataset_train.csv 

Show dataset:
    Histogram:
        python main.py -v Histogram ./datasets/dataset_train.csv
    Pair:
        python main.py -v Pair ./datasets/dataset_train.csv
    Scatter:
        By default:
            python main.py -v Scatter ./datasets/dataset_train.csv
        With specific features:
            python main.py -v Scatter ./datasets/dataset_train.csv  'Astronomy' 'Herbology'

train with eval:
    python main.py -t datasets/dataset_train.csv --eval

train to test:
    python main.py -t datasets/dataset_train.csv

predict test:
    python main.py -p datasets/dataset_test.csv

check predictions:
    python main.py -test houses.csv datasets/dataset_truth.csv 