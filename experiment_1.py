import os
import random
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mondrianforest import MondrianForestClassifier
from sklearn.metrics import accuracy_score, f1_score

RANDOM_SEED = 42
np.random.seed = RANDOM_SEED
random.seed = RANDOM_SEED

NUM_FEATURES = ["YEAR", "MONTH", "DAY_OF_MONTH", "DEP_TIME", "DISTANCE"]
CAT_FEATURES = ["UNIQUE_CARRIER", "ORIGIN", "DEST"]
FEATURES = NUM_FEATURES + CAT_FEATURES
Y = ["DEP_DEL15"]

AIRLINE_CODES = list(pd.read_csv("data/airlines.csv")["IATA_CODE"].values)
AIRPORT_CODES = list(pd.read_csv("data/airports.csv")["IATA_CODE"].values)

DIR = "data/AirOnTimeCSV"
filepaths = [os.path.join(DIR, filename) for filename in os.listdir(DIR)]

N_TREES=80
OFFLINE_N_JOBS = 8
offline_forest = RandomForestClassifier(n_estimators=N_TREES, n_jobs=OFFLINE_N_JOBS)
online_forest = MondrianForestClassifier(n_tree=N_TREES)

def load_and_preprocess(path, batch_size_limit=None):
    df = pd.read_csv(path)
    if batch_size_limit:
        _df0 = df[df[Y[0]]==0.0].sample(batch_size_limit//2)
        _df1 = df[df[Y[0]]==1.0].sample(batch_size_limit//2)
        df = pd.concat([_df0, _df1], axis=0)
    df = df[FEATURES+Y].dropna().reset_index(drop=True)
    airline_dummies = pd.get_dummies(df["UNIQUE_CARRIER"].astype(pd.CategoricalDtype(AIRLINE_CODES)), prefix="airline")
    origin_dummies = pd.get_dummies(df["ORIGIN"].astype(pd.CategoricalDtype(AIRPORT_CODES)), prefix="orig")
    dest_dummies = pd.get_dummies(df["DEST"].astype(pd.CategoricalDtype(AIRPORT_CODES)), prefix="dest")
    return pd.concat([df[NUM_FEATURES+Y], airline_dummies, origin_dummies, dest_dummies], axis=1)


MAX_ITERS = 10
TRAIN_OFFLINE = True
DEBUG = True
BATCH_SIZE_LIMIT = 750

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

for i, filepath in enumerate(filepaths):
    new_batch = load_and_preprocess(filepath, batch_size_limit=BATCH_SIZE_LIMIT)
    new_X = new_batch.drop(columns=Y).values
    new_y = np.ravel(new_batch[Y].values)
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
        X_test = np.concatenate([x_test_sample, new_X_test_sample])
        y_test = np.concatenate([y_test_sample, new_y_test_sample])

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

    if DEBUG:
        print("=============")

    if i == MAX_ITERS - 1:
        break
