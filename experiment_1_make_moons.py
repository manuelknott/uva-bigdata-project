import random
import time
import argparse

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skgarden import MondrianForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from utils import get_unbalanced_moons

from guppy import hpy

parser = argparse.ArgumentParser()
parser.add_argument('n_iter', help='Number of iterations', type=int)
parser.add_argument('batch_size', help='Number of entries per batch', type=int)
parser.add_argument('--n_trees', help='Number of trees', default=40, type=int)
parser.add_argument('--max_depth', help='max tree depth', default=50, type=int)
parser.add_argument('--offline_only', help='If true, online training is skipped.', action='store_true')
parser.add_argument('--online_only', help='If true, offline training is skipped.', action='store_true')
args = parser.parse_args()

RANDOM_SEED = 42
np.random.seed = RANDOM_SEED
random.seed = RANDOM_SEED

N_TREES = args.n_trees
MAX_DEPTH = args.max_depth
OFFLINE_N_JOBS = 1
offline_forest = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH, n_jobs=OFFLINE_N_JOBS)
online_forest = MondrianForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH)

MAX_ITERS = args.n_iter
TRAIN_OFFLINE = not args.online_only
TRAIN_ONLINE = not args.offline_only
DEBUG = False
BATCH_SIZE_LIMIT = args.batch_size

offline_X_train = None
offline_y_train = None
X_test = None
y_test = None

offline_accuracy = []
offline_f1score = []
online_accuracy = []
online_f1score = []
offline_training_times = []
online_update_times = []

_iters_phase_1 = args.n_iter // 2
_iters_phase_2 = args.n_iter - _iters_phase_1

balance_fractions = [0.25] * _iters_phase_1 + [0.75] * _iters_phase_2

for i, balance_fraction in enumerate(balance_fractions):
    new_X, new_y = get_unbalanced_moons(BATCH_SIZE_LIMIT, balance_fraction=balance_fraction)
    new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(new_X, new_y)

    if i == 0:
        offline_X_train = new_X_train
        offline_y_train = new_y_train
        X_test = new_X_test
        y_test = new_y_test
    else:
        old_ratio = i / (i + 1)
        offline_X_train_sample, _, offline_y_train_sample, _ = train_test_split(offline_X_train,
                                                                                offline_y_train,
                                                                                train_size=old_ratio)
        x_test_sample, _, y_test_sample, _ = train_test_split(X_test, y_test, train_size=old_ratio)

        new_X_train_sample, _, new_y_train_sample, _ = train_test_split(new_X_train, new_y_train,
                                                                        train_size=1 - old_ratio)
        new_X_test_sample, _, new_y_test_sample, _ = train_test_split(new_X_test, new_y_test,
                                                                      train_size=1 - old_ratio)

        offline_X_train = np.concatenate([offline_X_train_sample, new_X_train_sample])
        offline_y_train = np.concatenate([offline_y_train_sample, new_y_train_sample])

        X_test = new_X_test
        y_test = new_y_test

    if DEBUG:
        print(f"iter: {i}")
        print(f"offline_X_train size: {len(offline_X_train)}")
        print(f"offline_y_train size: {len(offline_y_train)}")
        print(f"X_test size: {len(X_test)}")
        print(f"y_test size: {len(y_test)}")

    if TRAIN_OFFLINE:
        if DEBUG:
            print("Fitting offline.")
        _time = time.time()
        offline_forest.fit(offline_X_train, offline_y_train)
        _time = time.time() - _time
        offline_predictions = offline_forest.predict(X_test)
        offline_accuracy.append(accuracy_score(y_test, offline_predictions))
        offline_f1score.append(f1_score(y_test, offline_predictions))
        offline_training_times.append(_time)

        file_object = open('1_offline_results.txt', 'a')
        file_object.write(
            f'{accuracy_score(y_test, offline_predictions)} {f1_score(y_test, offline_predictions)} {_time}\n')
        file_object.close()

    if TRAIN_ONLINE:
        if DEBUG:
            print("Fitting online.")
        _time = time.time()
        online_forest.partial_fit(new_X_train, new_y_train)
        _time = time.time() - _time
        if DEBUG:
            print(f"online classes: {online_forest.classes}")

        online_probabilities = online_forest.predict_proba(X_test)
        online_predictions = online_probabilities.argmax(axis=1)
        online_accuracy.append(accuracy_score(y_test, online_predictions))
        online_f1score.append(f1_score(y_test, online_predictions))
        online_update_times.append(_time)

        # Open a file with access mode 'a'
        file_object = open('1_online_results.txt', 'a')
        file_object.write(f'{accuracy_score(y_test, online_predictions)} {f1_score(y_test, online_predictions)} {_time}\n')
        file_object.close()

    if i == MAX_ITERS - 1:
        break
