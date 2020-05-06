export TRAINING_DATA=input/train_preprocessed.csv
export TEST_DATA=input/test_preprocessed.csv
export MODEL=$1

FOLD=0 python -m script.train
FOLD=1 python -m script.train
FOLD=2 python -m script.train
FOLD=3 python -m script.train
FOLD=4 python -m script.train

# python -m script.predict