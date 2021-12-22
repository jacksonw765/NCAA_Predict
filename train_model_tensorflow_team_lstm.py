import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2', 'minutes_played', 'minutes_played_2']

team_list2020 = pd.read_csv('team_list_scores2020.csv')
team_list2021 = pd.read_csv('team_list_scores2021.csv')
team_list2022 = pd.read_csv('team_list_scores2022.csv')
master_df = pd.concat([team_list2022, team_list2021, team_list2020])
#master_df = pd.read_csv('team_list_scores2022.csv')
y = master_df['points_for'].to_numpy()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2',
                                    'points_for', 'points_against'], errors='ignore')
master_df = master_df.drop(columns=columns_drop, errors='ignore')
#X = np.round(master_df.to_numpy(), 2).tolist()
X = master_df.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
#y_train = y_train.reshape(1, y_train.shape[0])
#y_test = y_test.reshape(1, y_test.shape[0])
model = tf.keras.Sequential([
    #tf.keras.layers.LSTM(69, input_shape=(X_train.shape[0], 1, X_train.shape[2]), return_sequences=False),
    tf.keras.layers.LSTM(145, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    tf.keras.layers.SimpleRNN(512, return_sequences=True, activation='relu'),
    tf.keras.layers.LSTM(1000, return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    #tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(286, ),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(572),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    #
    # tf.keras.layers.Dense(1430, ),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),

    # tf.keras.layers.Dense(572,),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.4),
    #
    # tf.keras.layers.Dense(200, ),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    #
    # tf.keras.layers.Dense(100,),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    #
    # tf.keras.layers.Dense(50, ),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.4),

    # tf.keras.layers.Dense(2860, ),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    #
    # tf.keras.layers.Dense(1430, ),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    #
    # tf.keras.layers.Dense(572),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.4),

    # tf.keras.layers.Dense(286,),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1, ),
])

#model.compile(loss='mse', optimizer='adadelta')
model.compile(loss='mse', optimizer='rmsprop',)
#rmsprop

model.fit(X_train, y_train, validation_data=(X_test,y_test), steps_per_epoch=100, epochs=256)
json_file = model.to_json()
with open('lstm.json', "w") as file:
    file.write(json_file)
model.save_weights('lstm.h5')
print('model trained')

# output = team1 + ": " + str(simulate_game(team1, predict_X_1))
# output2 = team2 + ": " + str(simulate_game(team2, predict_X_2))
# print(output)
# print(output2)

