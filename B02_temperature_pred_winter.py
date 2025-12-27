import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt

import pickle

Last_step_temp_to_used = 12
num_intervals_to_pred = 12
temp_data = pd.read_csv('data/temperature_final_5_min_winter.csv')
X = []
Y_dict = {}
for i in range(len(temp_data)):
    if i >= Last_step_temp_to_used:
        X.append(temp_data.iloc[i - Last_step_temp_to_used:i, 1].copy())
        for next in range(num_intervals_to_pred):
            if next not in Y_dict:
                Y_dict[next] = []
            if i+next < len(temp_data):
                Y_dict[next].append(temp_data.iloc[i+next, 1].copy())



X = np.array(X)
for next in range(num_intervals_to_pred):
    Y_dict[next] = np.array(Y_dict[next])

num_train = int(len(X) / 2)
X_train = X[:num_train]
X_test = X[num_train:]
Y_train_dict = {}
Y_test_dict = {}

for next in range(num_intervals_to_pred):
    Y_train_dict[next] = Y_dict[next][:num_train]
    Y_test_dict[next] = Y_dict[next][num_train:]


models = {}


# future things:
# add information / features about time of a day.
# add historical temp at the same time in last day.


for i in range(num_intervals_to_pred):
    model_name = f'next_{i*5}_{(i+1)*5}_min'
    print(X_train.shape)
    print(Y_train_dict[i].shape)
    reg = LinearRegression().fit(X_train, Y_train_dict[i])

    models[model_name] = reg
    ########
    X_test = X_test[:len(Y_test_dict[i])]
    print(X_test.shape)
    print(Y_test_dict[i].shape)
    Y_test_pred = reg.predict(X_test)

    print('R sq test', model_name, reg.score(X_test, Y_test_dict[i]))

    # num_sample_to_look = 15

    # plt.plot(range(num_sample_to_look), Y_test[:num_sample_to_look], 'r', label = 'Actual Temp')
    # plt.plot(range(num_sample_to_look), Y_test_pred[:num_sample_to_look], 'b', label = 'Pred Temp')
    # plt.legend()
    # plt.show()


with open(r"model/Temperature_predictor_winter.pickle", "wb") as output_file:
    pickle.dump(models, output_file)