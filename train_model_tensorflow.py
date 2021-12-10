import statistics
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.applications.densenet import layers
from keras.optimizer_v2.adam import Adam
from keras.saving.model_config import model_from_json
from sklearn.model_selection import train_test_split
from sportsipy.ncaab.teams import Teams
from sportsipy.ncaaf.teams import Teams as Teamsf
from sklearn.datasets import load_iris
from sportsipy.ncaab.boxscore import Boxscore
from sportsipy.ncaaf.schedule import Schedule
from sportsipy.ncaaf.boxscore import Boxscore as Boxscoref
from sklearn.model_selection import train_test_split

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2',]


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


master_df = pd.read_csv('team_list_scores2022_w.csv')
y = master_df['points_for'].to_numpy().tolist()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2', 'points_for', 'points_against'])
master_df = master_df.drop(columns=columns_drop)
#X = np.round(master_df.to_numpy(), 2).tolist()
X = master_df.to_numpy().tolist()


X_train, X_test, y_train, y_test = train_test_split(X, y)
model = tf.keras.Sequential([
    tf.keras.layers.Input(147),

    tf.keras.layers.Dense(286, ),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(572),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(1430, ),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(2860, ),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(5720, ),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(2860, ),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1430, ),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(572),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(286,),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, ),
])

#model.compile(loss='mse', optimizer='adadelta')
model.compile(loss='mse', optimizer='adam', metrics=['accuracy', f1_metric])

model.fit(X_train, y_train, validation_data=X_test, steps_per_epoch=100, epochs=512)
json_file = model.to_json()
with open('new_bad_model', "w") as file:
   file.write(json_file)
model.save_weights('new_bad_model.h5')
print('model trained')

# output = team1 + ": " + str(simulate_game(team1, predict_X_1))
# output2 = team2 + ": " + str(simulate_game(team2, predict_X_2))
# print(output)
# print(output2)

# 3.1 Save The Model
# # serialize to JSON
# json_file = model.to_json()
# with open(json_file_path, "w") as file:
#    file.write(json_file)
# # serialize weights to HDF5
# model.save_weights(h5_file)
# 3.2 Load The Model
# from keras.models import model_from_json
# # load json and create model
# file = open(json_file, 'r')
# model_json = file.read()
# file.close()
# loaded_model = model_from_json(model_json)
# # load weights
# loaded_model.load_weights(h5_file)


